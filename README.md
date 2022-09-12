# escape-room
* "Robots" in a 2D room are trained to escape from the door, using RL.
* Each robot has a "Lidar", which allows it to sense the distance to the nearest object in any line of sight, and the object type (wall, door, robot).
* Each robot can turn, increase and decrease its speed.
* Many experiments can be done: different room geometries; robots with shared policies; communication between robots.
* In the current setting, all robots share the same policy, but do not share the state or actions. i.e they do not directly communicate, there is no centralized decision making, but all robots use the same logic to make decisions.
* This video exemplifies the state seen by a robot:

![vid_state](https://user-images.githubusercontent.com/5672933/189739622-4b356189-2cb6-4ab5-b75d-f8572be7151a.gif)


* This video shows the performance during training using DQN, at various time steps along the training:

![vid_1](https://user-images.githubusercontent.com/5672933/189739596-f6972d26-bdc4-48c7-8229-b241f8f1c051.gif)

![vid_2](https://user-images.githubusercontent.com/5672933/189739603-354e905c-2691-46e3-99b1-12cd3a03f271.gif)

![vid_3](https://user-images.githubusercontent.com/5672933/189739609-d829c0a4-10eb-4455-80c3-55d84df142e9.gif)

![vid_4](https://user-images.githubusercontent.com/5672933/189739616-665b12a5-510a-447d-8fcb-e8562307dc6c.gif)
