from setuptools import setup

package_name = 'hand_landmarker'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='YOUR_NAME',
    maintainer_email='you@example.com',
    description='Hand landmarker node',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'hand_landmarker_node = hand_landmarker.hand_landmarker_node:main',
        ],
    },
)
