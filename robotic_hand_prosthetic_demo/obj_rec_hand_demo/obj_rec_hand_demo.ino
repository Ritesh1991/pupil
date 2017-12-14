
/*
Control an OpenBionics prosthetic hand with Arduino Mega 2560 using real-time information from ROS.  
This script subscribes to a rostopic with labels of detected objects and performs predetermined 
grasps based on those labels.
I.e., if a banana is detected, a banana grasp will form on the prosthetic hand.

Pin Layout on Arduino Mega 2560 is as follows:
Potentiometer feedback from linear actuators: [thumb, pinky, ring, middle, index] -> [A0,A1,A2,A3,A4]
Digital Outputs to each [M+, M-] pair on linear actuators: 
[thumb, pinky, ring, middle, index] -> [ [13,12], [11,10], [9,8], [7,6], [5,4] ]

Future edits: 
1) Get parallel control of fingers working correctly.
2) Implement force sensors on finger tips and EMG functionality for more user control

Author: Jesse Weisberg
*/

#if (ARDUINO >= 100)
  #include <Arduino.h>
#else
  #include <WProgram.h>
#endif
#include <StandardCplusplus.h>
#include <vector>
#include <map>
#include <alloca.h>
#include <string>
#include <cstring>
#include <iostream>

#include <ros.h>
#include <std_msgs/String.h>

using namespace std;

std::map< int, std::vector<int> > finger_map; //maps fingers to pin numbers
std::map< std::string, std::vector< vector <int>> > grip; //maps objects to predetermined grips

ros::NodeHandle nh;
std::string label;
boolean go;

// A publisher, if you want to publish anything back for testing/other purposes
//std_msgs::String msg;
//ros::Publisher label_pub("object_detection_label_feedback", &msg);

// Label callback function
void label_cb( const std_msgs::String& label_new ){
  go=true;
  label = label_new.data;  
}

// Instantiate subscriber that receives label of object_detection
ros::Subscriber<std_msgs::String> label_sub("object_detection_label", label_cb);

void setup()
{
  Serial.begin(57600); 
  for(int i=13; i>=2; i--){
    pinMode(i,OUTPUT);
  }
  //[thumb, pinky, ring, middle, index] -> [0,1,2,3,4]
  finger_map[0].push_back(13);  finger_map[0].push_back(12);
  finger_map[1].push_back(11);  finger_map[1].push_back(10);
  finger_map[2].push_back(9);   finger_map[2].push_back(8);
  finger_map[3].push_back(7);   finger_map[3].push_back(6);
  finger_map[4].push_back(5);   finger_map[4].push_back(4);

  //different grasp types
  const vector <int> cup = {700, 500, 500, 500, 450}; 
  //const vector <int> pen = {11, 22, 10, 321, 191}; 
  const vector <int> banana = {1003, 200, 200, 200, 50}; 
  //const vector <int> cap = {10, 921, 921, 921, 274}; 
  const vector <int> contract = {0, 0, 0, 0, 0};
  const vector <int> extend = {1000, 1000, 1000, 1000, 1000};
  const vector <int> orange = {300, 500, 400, 550, 400};
  const vector <int> book = {0, 0, 0, 0, 1000};
  
  grip["cup"].push_back(cup); 
  //grip["pen"].push_back(pen);
  grip["banana"].push_back(banana); 
  //grip["cap"].push_back(cap); 
  grip["contract"].push_back(contract); 
  //grip["laptop"].push_back(contract);
  grip["extend"].push_back(extend); 
  grip["orange"].push_back(orange);
  grip["apple"].push_back(orange);
  grip["book"].push_back(book);
  grip["cellphone"].push_back(banana);

  go = true;
  nh.initNode();
  nh.subscribe(label_sub);
  //nh.advertise(label_pub);
}

// Still working on this script, I apologize for the sloppiness.  
// Finger is the index of the finger in finger_map, pos is the desired position // constant control
// 1024: fully extended, 0: fully closed
//void move_finger(const int& finger, const int& pos){   
//  int pin1=LOW; int pin2=HIGH;
//  //if (analogRead(finger) > pos){pin1 = LOW;  pin2 = HIGH;}  //contract config
//  if (analogRead(finger) < pos){pin1 = HIGH; pin2 = LOW;} //extend config
//
//  digitalWrite(finger_map[finger][0], pin1);
//  digitalWrite(finger_map[finger][0]-1, pin2);
//}

void move_finger(int finger, const int& pos, const std::string& label){   
  int pin1=LOW; int pin2=HIGH;
  //if (analogRead(finger) > pos){pin1 = LOW;  pin2 = HIGH;}  //contract config
  if (analogRead(finger) < pos){pin1 = HIGH; pin2 = LOW;} //extend config
  
  Serial.println();
  Serial.print(finger);  Serial.print(" Desired Position Initial: "); Serial.print(grip[label][0][finger]); //Serial.println();
  Serial.print(" Actual Position Initial: ");  Serial.print(analogRead(finger));  Serial.println();
  //Serial.println(abs(grip[label][0][finger] - analogRead(finger)));
    
  while(abs(grip[label][0][finger] - analogRead(finger)) > 50) {
    //Serial.println(analogRead(finger));
    digitalWrite(finger_map[finger][0], pin1);
    digitalWrite(finger_map[finger][1], pin2);
  }
  stop_finger(finger);
}

void stop_finger(const int& finger){
  digitalWrite(finger_map[finger][0], LOW);
  digitalWrite(finger_map[finger][0]-1, LOW);
}

void stop_hand(){
  for(int i=13; i>=4; i--){
    digitalWrite(i,LOW);
  }
}

// Moves fingers 1 by 1
void move_hand(const std::string& label){ 
  for (int i=0; i<5; i++){  
    move_finger(i, grip[label][0][i], label);
    Serial.print(i); Serial.print(" Desired Position Final: "); Serial.print(grip[label][0][i]); 
    Serial.print(" Actual Position Final: "); Serial.print(analogRead(i)); Serial.println();
  }  
  delay(5000);
}

// Moves all fingers in parallel instead of one by one
//void move_hand(const std::string& label){
//  int temp=1;
//  int counter=0;
//  while(temp==1){   
//    for (int i=0; i<5; i++){
//      move_finger(i, grip[label][0][i]);
//      Serial.print(i); Serial.print(" Desired Position: "); Serial.print(grip[label][0][i]); 
//      Serial.print(" Actual Position: "); Serial.print(analogRead(i)); Serial.println();
//      if(std::abs(int(grip[label][0][i] - analogRead(i))) < 100) {
//        counter++;
//        stop_finger(i);
//      }
//    }  
//    //nh.spinOnce();
//    if(counter>=4) {
//      temp=0;
//    }
//  }
//  //delay(5000);
//}

void loop()
{ 
  if (go==true){
    digitalWrite(3, HIGH); // Turn on LED

    // For 
//    char *s3 = (char *)malloc(label.size() + 1);
//    memcpy(s3, label.c_str(), label.size() + 1);
//    msg.data = s3;
       
    if (!label.empty()){
      // Make sure received label is in the map, otherwise odd motion will occur.
      if ( grip.find(label) == grip.end() ){
        //std_msgs::String error; error.data = "Grip not found for detected object";
        //label_pub.publish(&error);
      } 
      else {
        //label_pub.publish(&msg);
        move_hand(label);
        //free(s3);
      }    
    }   
  }
  go=false;  
  
  digitalWrite(3, LOW);
  nh.spinOnce();
  delay(1);
}


