from flask import Flask, render_template, Response  # also , request
from smbus2 import SMBus, i2c_msg
from datetime import datetime
from flask_socketio import SocketIO, emit
import numpy as np
from tflite_runtime.interpreter import Interpreter
import cv2 as cv
import time
import board
import adafruit_veml7700
import busio
import pandas as pd
import csv
import serial

app = Flask(__name__)


modelDIR = "models"
modelPATH = modelDIR + "/" + "model.tflite"
labelPATH = modelDIR + "/" + "labels.txt"

df_tmp = "static" + '/' + "IsTemperature.csv"
df_light = "static" + '/' + "IsLight.csv"
df_rh = "static" + '/' + "IsHumidity.csv"
df_hcoh = "static" + '/' + "HCOH.csv"

interpreter = Interpreter(modelPATH)
interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]['shape']
down_points = (width, height)

DEV_ADDR_SHT4X = 0x44
BUS_ADDRESS = 1

#ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
#ser.reset_input_buffer()

def load_labels(path):
    f = open(path, "r")
    return [line.strip() for line in f]

def hcoh():
    ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
    ser.reset_input_buffer()
    index = 0
    while index < 1:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').rstrip()
            output = float(line)
            index = 1
    return output

def set_input_tensor(interpreter, input_data):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0] 
    input_tensor[:, :] = input_data


def classify(interpreter, input_data, top_k):
    set_input_tensor(interpreter, input_data)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0] 

    output_prob_q = np.squeeze(interpreter.get_tensor(output_details['index']))
    scale, zero_point = output_details['quantization']
    output_prob = scale * (output_prob_q - zero_point) 

    ordered_classes = np.argpartition(-output_prob, 1)

    return [(i, output_prob[i]) for i in ordered_classes[:top_k]][0]
    
def gen_7700():
	i2c = busio.I2C(board.SCL, board.SDA)
	veml7700 = adafruit_veml7700.VEML7700(i2c)
	print("Ambient light:", veml7700.light)
	time.sleep(1)
	light = veml7700.light
	return light

def gen_sht4x():
	with SMBus(1) as bus:
		write_data = i2c_msg.write(0x44, [0xFD])
		bus.i2c_rdwr(write_data)
		time.sleep(0.1)

		read_data = i2c_msg.read(0x44, 6)
		bus.i2c_rdwr(read_data)
		time.sleep(0.1)

		data_write = list(write_data)
		data_read = list(read_data)
#		print(data_write,data_read)
		t_1 = data_read[0]*256+data_read[1]
		t_deg = -45 + 175*t_1/65535
		t_deg = round(t_deg , 2)

		rh_1 = data_read[3] * 256 + data_read[4]
		rh_pRH = -6 + 125 * rh_1/65535
		time.sleep(1)
		return [t_deg, rh_pRH]
    
def gen_camera():
    camera = cv.VideoCapture(0) 
    file_type = ".jpg"
    
    while True:        
        ret, frame = camera.read()       
        ret, jpeg = cv.imencode(file_type, frame) 
        frame_bytes = jpeg.tobytes()        
        
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
                
        image_resized = cv.resize(frame, down_points, interpolation= cv.INTER_LINEAR)
        socketio.sleep(0.1)
        top_k = 1
        label_id, prob = classify(interpreter, image_resized, top_k)
        labels = load_labels(labelPATH)
        MLoutput_label = labels[label_id]
        message_content1 = 'Class #' + str(label_id) + ', Label: ' + MLoutput_label.split(" ")[1] + ', Probability: ' + str(np.round(prob*100, 2)) + '%'
        socketio.emit('ml_label',{'message': message_content1})
        
        socketio.sleep(0.1) 
        date = datetime.now()
        message_content = str(date)
        socketio.emit('datetime',{'message': message_content})
        
    camera.release() 
    
    
@app.route('/video_feed')
def video_feed():                                 
    return Response(gen_camera(),
            mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template("homepg.html")

@app.route("/about")
def about():
    return render_template("aboutpg.html")

@app.route("/contact")
def contact():
    return render_template("contactpg.html")

@app.route("/select")
def select():
    return render_template("selectpg.html")
@app.route('/room1')
def read_i2c_1():
	t_deg = 0
	rh_pRH = 0
	al = 0
	while True:
		[t_deg,rh_pRH] = gen_sht4x()
		al = gen_7700()
		hcoh_1 = hcoh()
		time.sleep(3)
		return render_template("room1.html", t_deg=t_deg, rh_pRH=rh_pRH, al = al, hcoh = hcoh_1 )

@app.route('/room2')
def read_i2c_2():
        t_deg = 0
        rh_pRH = 0
        al = 0
        while True:
                [t_deg,rh_pRH] = gen_sht4x()
                al = gen_7700()
                hcoh_1 = hcoh()
                time.sleep(3)
                return render_template("room2.html", t_deg=t_deg, rh_pRH=rh_pRH, al = al, hcoh = hcoh_1 )

@app.route('/room3')
def read_i2c_3():
        t_deg = 0
        rh_pRH = 0
        al = 0
        while True:
                [t_deg,rh_pRH] = gen_sht4x()
                al = gen_7700()
                hcoh_1 = hcoh()
                time.sleep(3)
                return render_template("room3.html", t_deg=t_deg, rh_pRH=rh_pRH, al = al, hcoh = hcoh_1 )

@app.route('/socket')
def socket():
    return render_template("index_socket.html")
    
@app.route('/ml')
def ml():
    return render_template("index_ml.html")

@app.route('/tphistory')
def Rec_temp1():
    [temp,rh_pRH] = gen_sht4x()
    Temperature_data = {
    'Time': [datetime.now().strftime("%Y/%m/%d %H:%M")],
    'Data': [temp]
    }
    df = pd.DataFrame(Temperature_data)
 
    df.to_csv(df_tmp, mode='a', index=False, header=False)
    
    data = []
    with open(df_tmp, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return render_template('Data_History.html', data=data, name = "Temperature")
    
@app.route('/lihistory')
def Rec_light1():
    lt = gen_7700() 
    Light_data = {
    'Time': [datetime.now().strftime("%Y/%m/%d %H:%M")],
    'Data': [lt]
    }
    df = pd.DataFrame(Light_data)
 
    df.to_csv(df_light, mode='a', index=False, header=False)
    
    data = []
    with open(df_light, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return render_template('Data_History.html', data=data, name = "Light")

@app.route('/rhhistory')
def Rec_humidity1():
    [temp,rh_pRH] = gen_sht4x()
    Humidity_data = {
    'Time': [datetime.now().strftime("%Y/%m/%d %H:%M")],
    'Data': [rh_pRH]
    }
    df = pd.DataFrame(Humidity_data)
 
    df.to_csv(df_rh, mode='a', index=False, header=False)
    
    data = []
    with open(df_rh, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return render_template('Data_History.html', data=data, name = "Humidity")
@app.route('/ch2ohistory')
def Rec_hcoh1():
    formaldehyde = hcoh()
    hcoh_data = {
    'Time': [datetime.now().strftime("%Y/%m/%d %H:%M")],
    'Data': [formaldehyde]
    }
    df = pd.DataFrame(hcoh_data)
 
    df.to_csv(df_hcoh, mode='a', index=False, header=False)
    
    data = []
    with open(df_hcoh, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return render_template('Data_History.html', data=data, name = "Formaldehyde")

if __name__ == '__main__':
    socketio = SocketIO(app)
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, use_reloader=False)
    
    
