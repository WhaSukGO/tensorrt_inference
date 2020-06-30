import engine as eng
import inference as inf
import tensorrt as trt 
import cv2
import numpy as np

input_file_path = "/home/whasukgo/FINAL/img/EVANS_TSITS_full.mp4"
onnx_file = "redaction.onnx"
serialized_plan_fp32 = "redaction.plan"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

engine = eng.load_engine(trt_runtime, serialized_plan_fp32)

print("=============================")
print("=Engine loaded successfully!=")
print("=============================")

HEIGHT = 360
WIDTH = 640
HEIGHT = 512
WIDTH = 864
#colors = [  ( i, i, i  ) for i in range(0, 256)  ]


def main():
	video()

def video():
	cap = cv2.VideoCapture(input_file_path)

	while cap.isOpened():

		_, frame = cap.read()

		resized = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		resized = cv2.resize(resized, (HEIGHT, WIDTH))
		resized = (2.0/255.0) * resized - 1.0
		resized = resized.transpose((2,0,1))

#		resized = cv2.resize(frame, (WIDTH, HEIGHT))
#		resized = resized.astype(np.float32)
#		resized = np.rollaxis(resized, 2, 0)		

		h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, 1, trt.float32)

		out = inf.do_inference(engine, resized, h_input, d_input, h_output, d_output, stream, 1, HEIGHT, WIDTH)

		#pr = out.reshape(( 360, 640 , 256 ) ).argmax( axis=2 )
		#pr = pr.astype(np.uint8)
	
#		cv2.imshow('frame',frame)

		print(out)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

if __name__ == "__main__":
	main()

