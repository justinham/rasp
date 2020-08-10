import tensorflow as tf
import cv2
import time
import argparse

import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.2)
parser.add_argument('--file', type=str, default="./vid/test5.mp4")
args = parser.parse_args()

max_number = 5

def parse_video():
    # with tf.compat.v1.Session() as sess:
    with tf.Session() as sess:
        ## config
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        ## load video file
        if args.file is not None:
            cap = cv2.VideoCapture(args.file)
        else:
            cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        # VideoReaderWriter
        driveoutfile = 'output/output.mp4'
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
       
        # fourcc = cv2.VideoWriter_fourcc('h','2','6', '4')
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video = cv2.VideoWriter(driveoutfile, fourcc, fps, (width, height))

        ## posenet
        start = time.time()
        frame_count = 0

        while True:
            input_image, display_image, output_scale = posenet.read_cap(cap, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs, feed_dict={'image:0': input_image})

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)

            ## rescale
            keypoint_coords *= output_scale

            # print (pose_scores)
            # print (keypoint_coords)
            
            # TODO this isn't particularly fast, use GL for drawing and display someday...
            
            # overlay_image = posenet.draw_skel_and_kp(display_image, pose_scores, keypoint_scores, keypoint_coords, min_pose_score=0.15, min_part_score=0.1)
            overlay_image = posenet.draw_keypoints(display_image, pose_scores, keypoint_scores, keypoint_coords)


            #### forehead & lips ##################################### 

            forehead_keypoints = []
            lip_keypoints = []
            keypoints_eyes = (keypoint_coords[:,1,:], keypoint_coords[:,2,:])
            keypoints_nose = keypoint_coords[:,0,:]
            COLOR_RED = (255,0, 0)
            COLOR_GR = (0,255,0)
            COLOR_BK = (0,0,0)
            
            ## show fps
            txt = frame_count / (time.time() - start)
            cv2.putText(overlay_image, "fps:"+str(txt), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_BK, 2)                
           
                
            for i in range(max_number):

                ## forhead location (up_left + right_down)
                eye_dis = abs(keypoints_eyes[0][i][1]-keypoints_eyes[1][i][1])
                fh_ul = (int(keypoints_eyes[1][i][1]-eye_dis*0.4), int(keypoints_eyes[0][i][0]-eye_dis*0.25))
                fh_rd = (int(keypoints_eyes[0][i][1]+eye_dis*0.5), int(keypoints_eyes[1][i][0]-eye_dis*1.5))
                forehead_keypoints.append([fh_ul, fh_rd])     
                # print(fh_ul, fh_rd)

                ## lip location
                lip_dis = abs(keypoints_eyes[0][i][1]-keypoints_eyes[1][i][1])*1.5
                lip_ul = (int(keypoints_eyes[1][i][1]), int(keypoints_eyes[0][i][0]+0.5*lip_dis))
                lip_rd = (int(keypoints_eyes[0][i][1]), int(keypoints_eyes[1][i][0]+lip_dis))
                lip_keypoints.append([lip_ul, lip_rd])

                if lip_keypoints[i][0][1]==0:
                    continue
            
                cv2.rectangle(overlay_image, forehead_keypoints[i][0], forehead_keypoints[i][1], COLOR_GR, 2)
                txt = 'forhead'
                cv2.putText(overlay_image, txt, (forehead_keypoints[i][1][0], forehead_keypoints[i][1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GR, 2)
            
                cv2.rectangle(overlay_image, lip_keypoints[i][0], lip_keypoints[i][1], COLOR_RED, 2)
                txt = 'lips'
                cv2.putText(overlay_image, txt, (lip_keypoints[i][1][0], lip_keypoints[i][1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)                
                
            
            
            cv2.imshow('posenet', overlay_image)
            # video.write(overlay_image)


            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print('Average FPS: ', frame_count / (time.time() - start))

        video.release()
        cap.release()


if __name__ == "__main__":
    parse_video()