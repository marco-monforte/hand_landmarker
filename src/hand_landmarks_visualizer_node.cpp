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

static const float MAX_TOUCH = 800.0;

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

    void feedbackCallback(const hand_msgs::msg::RH56DFTPFeedback::SharedPtr msg)
    {
        std::string hand = parseHand(msg->hand_id);
        if(hand.empty()) return;

        auto &fb = feedback_[hand];

        std::map<std::string,float> raw{
            {"pinky_tip", maxVec(std::vector<float>(msg->pinky_tip_touch.begin(), msg->pinky_tip_touch.end()))},
            {"pinky_top", maxVec(std::vector<float>(msg->pinky_top_touch.begin(), msg->pinky_top_touch.end()))},
            {"pinky_palm", maxVec(std::vector<float>(msg->pinky_palm_touch.begin(), msg->pinky_palm_touch.end()))},
            {"ring_tip", maxVec(std::vector<float>(msg->ring_tip_touch.begin(), msg->ring_tip_touch.end()))},
            {"ring_top", maxVec(std::vector<float>(msg->ring_top_touch.begin(), msg->ring_top_touch.end()))},
            {"ring_palm", maxVec(std::vector<float>(msg->ring_palm_touch.begin(), msg->ring_palm_touch.end()))},
            {"middle_tip", maxVec(std::vector<float>(msg->middle_tip_touch.begin(), msg->middle_tip_touch.end()))},
            {"middle_top", maxVec(std::vector<float>(msg->middle_top_touch.begin(), msg->middle_top_touch.end()))},
            {"middle_palm", maxVec(std::vector<float>(msg->middle_palm_touch.begin(), msg->middle_palm_touch.end()))},
            {"index_tip", maxVec(std::vector<float>(msg->index_tip_touch.begin(), msg->index_tip_touch.end()))},
            {"index_top", maxVec(std::vector<float>(msg->index_top_touch.begin(), msg->index_top_touch.end()))},
            {"index_palm", maxVec(std::vector<float>(msg->index_palm_touch.begin(), msg->index_palm_touch.end()))},
            {"thumb_tip", maxVec(std::vector<float>(msg->thumb_tip_touch.begin(), msg->thumb_tip_touch.end()))},
            {"thumb_top", maxVec(std::vector<float>(msg->thumb_top_touch.begin(), msg->thumb_top_touch.end()))},
            {"thumb_middle", maxVec(std::vector<float>(msg->thumb_middle_touch.begin(), msg->thumb_middle_touch.end()))},
            {"thumb_palm", maxVec(std::vector<float>(msg->thumb_palm_touch.begin(), msg->thumb_palm_touch.end()))},
            {"palm", maxVec(std::vector<float>(msg->palm_touch.begin(), msg->palm_touch.end()))}
        };

        for(auto &kv : raw)
        {
            fb[kv.first] = ema_alpha_ * kv.second + (1 - ema_alpha_) * fb[kv.first];
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

    void render()
    {
        double t = now().seconds();

        for(auto &h : {"left","right"})
        {
            if(last_landmark_time_.find(h) != last_landmark_time_.end())
            {
                if(t - last_landmark_time_[h] > landmark_timeout_)
                    landmarks_[h].reset();
            }
        }

        cv::Mat canvas = latest_image_.empty() ? cv::Mat::zeros(720,1280,CV_8UC3) : latest_image_.clone();

        std::vector<std::string> hands = controlled_hand_ == "both" ? std::vector<std::string>{"left","right"} : std::vector<std::string>{controlled_hand_};

        for(auto &h : hands)
        {
            auto msg = landmarks_[h];
            if(!msg) continue;
            if(msg->landmarks.size() != 21) continue;

            auto pts = computePoints(msg->landmarks, canvas);
            drawHand(canvas, pts, h);
        }

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