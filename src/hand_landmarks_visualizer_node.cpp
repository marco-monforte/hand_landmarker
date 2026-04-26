#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <hand_msgs/msg/hand_landmarks.hpp>
#include <hand_msgs/msg/rh56_dftp_feedback.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <map>
#include <string>
#include <vector>
#include <algorithm>

using std::placeholders::_1;

static const float MAX_TOUCH = 200.0;

cv::Scalar getColor(float value, const std::string &hand)
{
    float v = std::max(0.f, std::min(value / MAX_TOUCH, 1.f));

    int r=0,g=0,b=0;

    if(hand == "left")
    {
        if(v < 0.5){
            r = int(2*v*255);
            g = 255;
        }else{
            r = 255;
            g = int((1-2*(v-0.5))*255);
        }
        b = 0;
    }
    else
    {
        if(v < 0.5){
            g = int(2*v*255);
            b = 255;
        }else{
            g = int((1-2*(v-0.5))*255);
            b = 255;
        }
        r = int(v*255);
    }

    return cv::Scalar(b,g,r);
}

class HandLandmarksVisualizer : public rclcpp::Node
{
public:

    HandLandmarksVisualizer() : Node("hand_landmarks_visualizer")
    {
        declare_parameter("controlled_hand","left");
        declare_parameter("ema_alpha",0.3);
        declare_parameter("landmark_timeout",0.1);
        declare_parameter("camera_topic","/camera/color/image_raw");
        declare_parameter("feedback_topic","/rh56dftp/feedback");

        controlled_hand_ = get_parameter("controlled_hand").as_string();
        ema_alpha_ = get_parameter("ema_alpha").as_double();
        landmark_timeout_ = get_parameter("landmark_timeout").as_double();

        std::string camera_topic = get_parameter("camera_topic").as_string();
        std::string feedback_topic = get_parameter("feedback_topic").as_string();

        image_sub_ = create_subscription<sensor_msgs::msg::Image>(
            camera_topic, 1, std::bind(&HandLandmarksVisualizer::imageCallback, this, _1));

        landmarks_sub_ = create_subscription<hand_msgs::msg::HandLandmarks>(
            "/hand_landmarks", 10, std::bind(&HandLandmarksVisualizer::landmarksCallback, this, _1));

        feedback_sub_ = create_subscription<hand_msgs::msg::RH56DFTPFeedback>(
            feedback_topic, 10, std::bind(&HandLandmarksVisualizer::feedbackCallback, this, _1));

        timer_ = create_wall_timer(
            std::chrono::milliseconds(30),
            std::bind(&HandLandmarksVisualizer::render, this));

        RCLCPP_INFO(get_logger(), "Hand visualizer started");
    }

private:

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<hand_msgs::msg::HandLandmarks>::SharedPtr landmarks_sub_;
    rclcpp::Subscription<hand_msgs::msg::RH56DFTPFeedback>::SharedPtr feedback_sub_;

    rclcpp::TimerBase::SharedPtr timer_;
    cv::Mat latest_image_;

    std::map<std::string, hand_msgs::msg::HandLandmarks::SharedPtr> landmarks_;
    std::map<std::string, std::map<std::string, float>> feedback_;
    std::map<std::string, double> last_landmark_time_;

    std::string controlled_hand_;
    double ema_alpha_;
    double landmark_timeout_;

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        latest_image_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
    }

    std::string parseHand(const std::string &id)
    {
        std::string lower = id;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        if(lower.find("left") != std::string::npos) return "left";
        if(lower.find("right") != std::string::npos) return "right";

        return "";
    }

    void landmarksCallback(const hand_msgs::msg::HandLandmarks::SharedPtr msg)
    {
        std::string hand = parseHand(msg->hand_id);
        if(hand.empty()) return;

        landmarks_[hand] = msg;
        last_landmark_time_[hand] = now().seconds();
    }

    float maxVec(const std::vector<float> &v)
    {
        if(v.empty()) return 0.0;
        return *std::max_element(v.begin(), v.end());
    }

    template<typename T>
    float maxArray(const T &arr)
    {
        if (arr.empty())
            return 0.0f;

        return static_cast<float>(*std::max_element(arr.begin(), arr.end()));
    }

    void feedbackCallback(const hand_msgs::msg::RH56DFTPFeedback::SharedPtr msg)
    {
        std::string hand = parseHand(msg->hand_id);

        if (hand.empty())
            return;

        auto &fb = feedback_[hand];

        std::map<std::string, float> raw{
            {"pinky_tip", maxArray(msg->pinky_tip_touch)},
            {"pinky_top", maxArray(msg->pinky_top_touch)},
            {"pinky_palm", maxArray(msg->pinky_palm_touch)},

            {"ring_tip", maxArray(msg->ring_tip_touch)},
            {"ring_top", maxArray(msg->ring_top_touch)},
            {"ring_palm", maxArray(msg->ring_palm_touch)},

            {"middle_tip", maxArray(msg->middle_tip_touch)},
            {"middle_top", maxArray(msg->middle_top_touch)},
            {"middle_palm", maxArray(msg->middle_palm_touch)},

            {"index_tip", maxArray(msg->index_tip_touch)},
            {"index_top", maxArray(msg->index_top_touch)},
            {"index_palm", maxArray(msg->index_palm_touch)},

            {"thumb_tip", maxArray(msg->thumb_tip_touch)},
            {"thumb_top", maxArray(msg->thumb_top_touch)},
            {"thumb_middle", maxArray(msg->thumb_middle_touch)},
            {"thumb_palm", maxArray(msg->thumb_palm_touch)},

            {"palm", maxArray(msg->palm_touch)}
        };

        for (auto &kv : raw)
        {
            fb[kv.first] = ema_alpha_ * kv.second + (1.0 - ema_alpha_) * fb[kv.first];
        }
    }

    std::vector<cv::Point> computePoints(
        const std::vector<geometry_msgs::msg::Pose> &lm,
        const cv::Mat &canvas)
    {
        int h = canvas.rows;
        int w = canvas.cols;

        std::vector<cv::Point> pts;
        for(const auto &p : lm)
        {
            pts.emplace_back(int(p.position.x * w), int(p.position.y * h));
        }

        return pts;
    }

    void drawHand(cv::Mat &canvas, const std::vector<cv::Point> &pts, const std::string &hand)
    {
        auto &fb = feedback_[hand];

        std::map<std::string,std::vector<int>> finger_map{
            {"pinky",{17,18,19,20}},
            {"ring",{13,14,15,16}},
            {"middle",{9,10,11,12}},
            {"index",{5,6,7,8}}
        };

        for(auto &f : finger_map)
        {
            auto id = f.second;
            cv::line(canvas, pts[id[0]], pts[id[1]], getColor(fb[f.first+"_palm"],hand), 3);
            cv::line(canvas, pts[id[1]], pts[id[2]], getColor(fb[f.first+"_top"],hand), 3);
            cv::line(canvas, pts[id[2]], pts[id[3]], getColor(fb[f.first+"_top"],hand), 3);

            int r = 8 + fb[f.first+"_tip"]/MAX_TOUCH*12;
            cv::circle(canvas, pts[id[3]], r, getColor(fb[f.first+"_tip"],hand), -1);
        }

        cv::line(canvas, pts[1], pts[2], getColor(fb["thumb_palm"],hand), 3);
        cv::line(canvas, pts[2], pts[3], getColor(fb["thumb_middle"],hand), 3);
        cv::line(canvas, pts[3], pts[4], getColor(fb["thumb_top"],hand), 3);

        int r = 8 + fb["thumb_tip"]/MAX_TOUCH*12;
        cv::circle(canvas, pts[4], r, getColor(fb["thumb_tip"],hand), -1);

        std::vector<std::pair<int,int>> palm{
            {0,1},{0,5},{5,9},{9,13},{13,17},{17,0}
        };
        for(auto &p : palm)
            cv::line(canvas, pts[p.first], pts[p.second], getColor(fb["palm"],hand), 2);

        cv::circle(canvas, pts[0], 7, cv::Scalar(255,255,255), -1);
    }

    void drawLegend(cv::Mat &canvas, int x0, int y0, const std::string &hand)
    {
        const int height = 300;
        const int width = 20;

        for (int i = 0; i < height; ++i)
        {
            float value = MAX_TOUCH * (1.0f - static_cast<float>(i) / height);

            cv::line(
                canvas,
                cv::Point(x0, y0 + i),
                cv::Point(x0 + width, y0 + i),
                getColor(value, hand),
                1);
        }

        cv::putText(
            canvas,
            std::to_string(static_cast<int>(MAX_TOUCH)),
            cv::Point(x0 - 10, y0 - 10),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(255, 255, 255),
            1);

        cv::putText(
            canvas,
            "0",
            cv::Point(x0 - 10, y0 + height + 18),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(255, 255, 255),
            1);
    }

    void drawLegends(cv::Mat &canvas)
    {
        int w = canvas.cols;

        if (controlled_hand_ == "left")
        {
            // Show only left legend
            drawLegend(canvas, 15, 70, "left");
        }
        else if (controlled_hand_ == "right")
        {
            // Show only right legend
            drawLegend(canvas, w - 35, 70, "right");
        }
        else if (controlled_hand_ == "both")
        {
            // Show both legends
            drawLegend(canvas, 15, 70, "left");
            drawLegend(canvas, w - 35, 70, "right");
        }
    }

    void render()
    {
        double t = now().seconds();

        for (const auto &h : {"left", "right"})
        {
            if (t - last_landmark_time_[h] > landmark_timeout_)
                landmarks_[h].reset();
        }

        cv::Mat canvas;

        if (latest_image_.empty())
            canvas = cv::Mat::zeros(720, 1280, CV_8UC3);
        else
            canvas = latest_image_.clone();

        std::vector<std::string> hands;

        if (controlled_hand_ == "both")
            hands = {"left", "right"};
        else
            hands = {controlled_hand_};

        for (const auto &h : hands)
        {
            auto msg = landmarks_[h];

            if (!msg)
                continue;

            if (msg->landmarks.size() != 21)
                continue;

            auto pts = computePoints(msg->landmarks, canvas);

            drawHand(canvas, pts, h);
        }

        // Draw color legends exactly like Python version
        drawLegends(canvas);

        cv::imshow("Hand Visualizer", canvas);
        cv::waitKey(1);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<HandLandmarksVisualizer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
