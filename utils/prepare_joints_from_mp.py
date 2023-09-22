import numpy as np

path = "/leonardo/home/usertrain/a08trb14/pose_landmarker_heavy.task"
image = '/leonardo/home/usertrain/a08trb14/MOTSynth/frames/000/rgb/0001.jpg'
annotations_path = '/leonardo/home/usertrain/a08trb14/annotations_clean/'
out_dir = '/leonardo/home/usertrain/a08trb14/kps_pixel_bbox/'
# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import sys

video_id = sys.argv[1]
image_id = 0
annotations = np.load(annotations_path + video_id.zfill(3) + ".npy")[:,0:5]



def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image






# Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path=path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)
table_out = []
# For each bbox
i_prev = np.nan
for i in range(annotations.shape[0]):
  if annotations[i,0] != i_prev:
    image_path = '/leonardo/home/usertrain/a08trb14/MOTSynth/frames/'+ video_id.zfill(3) +'/rgb/'+ str(int(annotations[i,0])).zfill(4) +'.jpg'
    image = mp.Image.create_from_file(image_path)
    i_prev = annotations[i,0]
    #print(i_prev)
  #print(annotations[i,:])
  cropped_image = image.numpy_view()[int(annotations[i,2]):int(annotations[i,4]), int(annotations[i,1]):int(annotations[i,3])]
  cv2.imwrite('contour1.png', cropped_image)
  cropped_image_mp = mp.Image(image_format=mp.ImageFormat.SRGB,data=cropped_image.copy())
  # STEP 4: Detect pose landmarks from the input image.
  detection_result = detector.detect(cropped_image_mp)
  if detection_result.pose_landmarks:
    #exit()
    img_height = int(annotations[i,3])-int(annotations[i,1])
    img_width = int(annotations[i,4])-int(annotations[i,2])
    #kps = [ [kp.x*img_width + int(annotations[i,1]),kp.y*img_height + int(annotations[i,2])] for kp in detection_result.pose_world_landmarks[0]]
    
    
    # kps = [ [kp.x*img_height+int(annotations[i,1]),kp.y*img_width+int(annotations[i,2])] for kp in detection_result.pose_landmarks[0]]
    kps = [ [kp.x*img_height,kp.y*img_width] for kp in detection_result.pose_landmarks[0]]
    kps = np.array(kps).reshape(1,-1)
    row = np.concatenate((np.array(i_prev).reshape((1,1)), kps),axis=1)
  else:
    row = np.zeros((1,67))
    row[0,0] = i_prev
    row = row.tolist()
  table_out.append(row)
np.save( out_dir + video_id.zfill(3) + ".npy",np.array(table_out).squeeze())

    # modified_image = cv2.circle(image.numpy_view(), (int(kps[0][0])+int(annotations[i,1]),int(kps[0][1])+int(annotations[i,2])), 2, [0,100,0], 2)
    
    # cv2.imwrite("abc.jpg",cv2.cvtColor(modified_image, cv2.COLOR_RGB2BGR))
    # exit()
  # # STEP 5: Process the detection result. In this case, visualize it.
  #annotated_image = draw_landmarks_on_image(cropped_image, detection_result)
  #cv2.imwrite("abc.jpg",cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
  

