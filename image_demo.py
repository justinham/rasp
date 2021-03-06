import tensorflow as tf
import cv2
import time
import argparse
import os

import posenet


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--output_dir', type=str, default='./output')
args = parser.parse_args()


def main():

    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.output_dir:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

        filenames = [
            f.path for f in os.scandir(args.image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]

        start = time.time()
        for f in filenames:
            input_image, draw_image, output_scale = posenet.read_imgfile(
                f, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.25)

            keypoint_coords *= output_scale

            ################################ justin add
            # print(keypoint_coords)
            forehead_keypoints = []
            lip_keypoints = []
            keypoints_eyes = (keypoint_coords[:,1,:], keypoint_coords[:,2,:])
            keypoints_nose = keypoint_coords[:,0,:]

            for i in range(10):

                ## forhead location (up_left + right_down)
                eye_dis = abs(keypoints_eyes[0][i][1]-keypoints_eyes[1][i][1])
                fh_ul = (int(keypoints_eyes[1][i][1]-eye_dis*0.4), int(keypoints_eyes[0][i][0]-eye_dis*0.25))
                fh_rd = (int(keypoints_eyes[0][i][1]+eye_dis*0.5), int(keypoints_eyes[1][i][0]-eye_dis*1.5))
                forehead_keypoints.append([fh_ul, fh_rd])     
                # print(fh1, fh2, fh3, fh4)

                ## lip location
                lip_dis = abs(keypoints_eyes[0][i][1]-keypoints_eyes[1][i][1])*1.5
                lip_ul = (int(keypoints_eyes[1][i][1]), int(keypoints_eyes[0][i][0]+0.5*lip_dis))
                lip_rd = (int(keypoints_eyes[0][i][1]), int(keypoints_eyes[1][i][0]+lip_dis))
                lip_keypoints.append([lip_ul, lip_rd])
                # print(lip1, lip2, lip3, lip4)
                
            #########################


            if args.output_dir:

                # draw_image = posenet.draw_skel_and_kp(draw_image, pose_scores, keypoint_scores, keypoint_coords, min_pose_score=0.25, min_part_score=0.25)

                ################### justin add
                COLOR_RED = (255,0, 0)
                COLOR_GR = (0,255,0)
                
                img = cv2.imread(f)
                draw_image = img.copy()
                    
                for i in range(10):

                    print("hh ", lip_keypoints[i])
                
                    if lip_keypoints[i][0][1]==0:
                        continue
                
                    cv2.rectangle(draw_image, forehead_keypoints[i][0], forehead_keypoints[i][1], COLOR_GR, 2)
                    txt = 'forhead'
                    cv2.putText(draw_image, txt, (forehead_keypoints[i][1][0], forehead_keypoints[i][1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GR, 2)
                
                    cv2.rectangle(draw_image, lip_keypoints[i][0], lip_keypoints[i][1], COLOR_RED, 2)
                    txt = 'lips'
                    cv2.putText(draw_image, txt, (lip_keypoints[i][1][0], lip_keypoints[i][1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)                
                
                # cv2.imshow("test", draw_image)  
                    
                ###############################
                
                cv2.imwrite(os.path.join(args.output_dir, os.path.relpath(f, args.image_dir)), draw_image)

            

            if not args.notxt:
                print("Results for image: %s" % f)
                for pi in range(len(pose_scores)):
                    if pose_scores[pi] == 0.:
                        break
                    print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                    for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                        print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))

        print('Average FPS:', len(filenames) / (time.time() - start))


if __name__ == "__main__":
    main()
