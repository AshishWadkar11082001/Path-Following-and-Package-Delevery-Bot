import  sys
import traceback
import time
import os
import math
from zmqRemoteApi import RemoteAPIClient
import zmq
import numpy as np
import cv2
import random
from pyzbar.pyzbar import decode

finish = 0
turn = 0
normal_speed = 0.5
stop = 0
distance = 0
cx = 0
cy = 0
x = 0
y = 0
z = 0
node = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'A']
a_clk = ['A', 'C', 'G', 'K', 'O']
clk = ['B', 'D', 'F', 'H', 'J', 'L', 'N', 'P']
drop_loc = ['E', 'I', 'M']

def read_image(sim):
	vision_sensor_handle = sim.getObjectHandle('./vision_sensor')
	img = sim.getVisionSensorImg(vision_sensor_handle, 0, 0)
	img = np.frombuffer(img[0], dtype = np.uint8)
	img1 = img.reshape(512,512,3)
	return img1

def activate_qr(q, sim):
		arena_dummy_handle = sim.getObject("/Arena_dummy") 

		## Retrieve the handle of the child script attached to the Arena_dummy scene object.
		childscript_handle = sim.getScript(sim.scripttype_childscript, arena_dummy_handle, "")

		## Call the activate_qr_code() function defined in the child script to make the QR code visible at checkpoints
		if( q == "E"):
			sim.callScriptFunction("activate_qr_code", childscript_handle, "checkpoint E")
		elif( q == "I"):
			sim.callScriptFunction("activate_qr_code", childscript_handle, "checkpoint I")
		else:
			sim.callScriptFunction("activate_qr_code", childscript_handle, "checkpoint M")

def deactivate_qr(q, sim):
		## Retrieve the handle of the Arena_dummy scene object.
		arena_dummy_handle = sim.getObject("/Arena_dummy") 

		## Retrieve the handle of the child script attached to the Arena_dummy scene object.
		childscript_handle = sim.getScript(sim.scripttype_childscript, arena_dummy_handle, "")

		## Call the deactivate_qr_code() function defined in the child script to make the QR code invisible at checkpoints
		if( q == "E"):
			sim.callScriptFunction("deactivate_qr_code", childscript_handle, "checkpoint E")
		elif( q == "I"):
			sim.callScriptFunction("deactivate_qr_code", childscript_handle, "checkpoint I")
		else:
			sim.callScriptFunction("deactivate_qr_code", childscript_handle, "checkpoint M")

def control_logic(sim):

	left_joint_handle = sim.getObjectHandle('./left_joint')
	right_joint_handle = sim.getObjectHandle('./right_joint')
	low = np.uint8([175,175,175])
	high = np.uint8([0,0,0])
	
	i = 0
	while i < 17:
		
		img = read_image(sim)
		def masking_image(img):
			low = np.uint8([175,175,175])
			high = np.uint8([0,0,0])
			img1 = cv2.inRange(img, high, low)
			return img1
		image = masking_image(img)
		contours, hierarchies = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		
		c = max(contours, key = cv2.contourArea)
		M = cv2.moments(c)
		cv2.drawContours(image, contours, -1, (0,255,0), 3)
		if M["m00"] != 0:
			
			if M["m00"] > 17000:
				cx = int(M['m10']/M['m00'])
				cy = int(M['m01']/M['m00'])

				if cx >= 270:
					sim.setJointTargetVelocity(left_joint_handle, 1)
					sim.setJointTargetVelocity(right_joint_handle, 0.3)
				if cx < 270 and cx > 250:
					sim.setJointTargetVelocity(left_joint_handle, 1.5)
					sim.setJointTargetVelocity(right_joint_handle, 1.5)
				if cx <= 250:
					sim.setJointTargetVelocity(left_joint_handle, 0.3)
					sim.setJointTargetVelocity(right_joint_handle, 1)

			elif M["m00"] > 12000:
				cx = int(M['m10']/M['m00'])
				cy = int(M['m01']/M['m00'])

				if cx >= 270:
					sim.setJointTargetVelocity(left_joint_handle, 0.5)
					sim.setJointTargetVelocity(right_joint_handle, 0)
				if cx < 270 and cx > 250:
					sim.setJointTargetVelocity(left_joint_handle, 0.5)
					sim.setJointTargetVelocity(right_joint_handle, 0.5)
				if cx <= 250:
					sim.setJointTargetVelocity(left_joint_handle, 0)
					sim.setJointTargetVelocity(right_joint_handle, 0.5)

			else:
				cx = int(M['m10']/M['m00'])
				cy = int(M['m01']/M['m00'])

				if cx >= 270:
					sim.setJointTargetVelocity(left_joint_handle, 0.4)
					sim.setJointTargetVelocity(right_joint_handle, -0.2)
				if cx < 270 and cx > 250:
					sim.setJointTargetVelocity(left_joint_handle, 0.5)
					sim.setJointTargetVelocity(right_joint_handle, 0.5)
				if cx <= 250:
					sim.setJointTargetVelocity(left_joint_handle, -0.2)
					sim.setJointTargetVelocity(right_joint_handle, 0.4)
			cv2.circle(image, (cx, cy), 5, (255,255,255), -1)


		if len(contours) > 19:
			time.sleep(0.2)
			current_node = node[i]
			print("Current Checkpoint:  ",current_node)
			
			sim.setJointTargetVelocity(left_joint_handle, 0)
			sim.setJointTargetVelocity(right_joint_handle, 0)
			time.sleep(1)			
			
			if current_node in a_clk:
				
				sim.setJointTargetVelocity(left_joint_handle, 1)
				sim.setJointTargetVelocity(right_joint_handle, 1)
				time.sleep(1.1)
				sim.setJointTargetVelocity(left_joint_handle, 0)
				sim.setJointTargetVelocity(right_joint_handle, 0)
				time.sleep(0.2)
				sim.setJointTargetVelocity(left_joint_handle, -1)
				sim.setJointTargetVelocity(right_joint_handle, 1)
				time.sleep(1.2)

			elif current_node in clk:
				sim.setJointTargetVelocity(left_joint_handle, 1)
				sim.setJointTargetVelocity(right_joint_handle, 1)
				time.sleep(1.1)
				sim.setJointTargetVelocity(left_joint_handle, 0)
				sim.setJointTargetVelocity(right_joint_handle, 0)
				time.sleep(0.2)
				
				sim.setJointTargetVelocity(left_joint_handle, 1)
				sim.setJointTargetVelocity(right_joint_handle, -1)
				time.sleep(1.2)
			
			else:
				if current_node in drop_loc:
					q = current_node
					check = 'checkpoint '+q
					activate_qr(q, sim)
					message = read_qr_code(sim)
					print(message)
					##DELIVER PACKAGE AT CURRENT CHECKPOINT
					if message == "Orange Cone":
						## Retrieve the handle of the Arena_dummy scene object.
						arena_dummy_handle = sim.getObject("/Arena_dummy") 

						## Retrieve the handle of the child script attached to the Arena_dummy scene object.
						childscript_handle = sim.getScript(sim.scripttype_childscript, arena_dummy_handle, "")

						## Deliver package_1 at checkpoint
						sim.callScriptFunction("deliver_package", childscript_handle, "package_1" , check)

					elif message == "Blue Cylinder":
						## Retrieve the handle of the Arena_dummy scene object.
						arena_dummy_handle = sim.getObject("/Arena_dummy") 

						## Retrieve the handle of the child script attached to the Arena_dummy scene object.
						childscript_handle = sim.getScript(sim.scripttype_childscript, arena_dummy_handle, "")

						## Deliver package_2 at checkpoint
						sim.callScriptFunction("deliver_package", childscript_handle, "package_2", check)

					else:
						## Retrieve the handle of the Arena_dummy scene object.
						arena_dummy_handle = sim.getObject("/Arena_dummy") 

						## Retrieve the handle of the child script attached to the Arena_dummy scene object.
						childscript_handle = sim.getScript(sim.scripttype_childscript, arena_dummy_handle, "")

						## Deliver package_3 at checkpoint
						sim.callScriptFunction("deliver_package", childscript_handle, "package_3", check)

					deactivate_qr(q, sim)

				sim.setJointTargetVelocity(left_joint_handle, 1)
				sim.setJointTargetVelocity(right_joint_handle, 1)
				time.sleep(1)
			
			i = i+1
			time.sleep(1)

def read_qr_code(sim):

	qr_message = None
	image = read_image(sim)
	image = read_image(sim)
	qrcode = decode(image)
	qr_message = qrcode[0].data.decode('utf-8')		

	return qr_message


if __name__ == "__main__":
	client = RemoteAPIClient()
	sim = client.getObject('sim')	

	try:

		## Start the simulation using ZeroMQ RemoteAPI
		try:
			return_code = sim.startSimulation()
			if sim.getSimulationState() != sim.simulation_stopped:
				print('\nSimulation started correctly in CoppeliaSim.')
			else:
				print('\nSimulation could not be started correctly in CoppeliaSim.')
				sys.exit()

		except Exception:
			print('\n[ERROR] Simulation could not be started !!')
			traceback.print_exc(file=sys.stdout)
			sys.exit()

		## Runs the robot navigation logic written by participants
		try:
			control_logic(sim)
			time.sleep(5)
			
		except Exception:
			print('\n[ERROR] Your control_logic function throwed an Exception, kindly debug your code!')
			print('Stop the CoppeliaSim simulation manually if required.\n')
			traceback.print_exc(file=sys.stdout)
			print()
			sys.exit()

		
		## Stop the simulation using ZeroMQ RemoteAPI
		try:
			return_code = sim.stopSimulation()
			time.sleep(0.5)
			if sim.getSimulationState() == sim.simulation_stopped:
				print('\nSimulation stopped correctly in CoppeliaSim.')
			else:
				print('\nSimulation could not be stopped correctly in CoppeliaSim.')
				sys.exit()

		except Exception:
			print('\n[ERROR] Simulation could not be stopped !!')
			traceback.print_exc(file=sys.stdout)
			sys.exit()

	except KeyboardInterrupt:
		## Stop the simulation using ZeroMQ RemoteAPI
		return_code = sim.stopSimulation()
		time.sleep(0.5)
		if sim.getSimulationState() == sim.simulation_stopped:
			print('\nSimulation interrupted by user in CoppeliaSim.')
		else:
			print('\nSimulation could not be interrupted. Stop the simulation manually .')
			sys.exit()