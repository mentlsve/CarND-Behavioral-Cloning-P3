AWS GPU Instance
===

#### Launching the instance

For training the model I am using an `g2.2xlarge` instance on AWS. It is running an the community AMI: `udacity-carnd - ami-52cd123d`. After starting the instance in the he AWS console (Actions -> Instance State -> Start) write down the public IP.

#### Connecting to the instance

Open a terminal on the local workstation and ssh into the instance: `ssh carnd@<public IP>`. The password is `carnd`.

#### Copying files to the instance

The training data is created on the local workstation by running the simulator. The output of the simulator runs are frames/images (the features) and the steering angle (the labels). These have to be uploaded to the instance

`scp data.zip carnd@<public IP>`