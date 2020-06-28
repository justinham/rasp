# PoseNet python sample program
import tensorflow as tf
import cv2
import time
import argparse
import os
import posenet

print('INIT:')
# from google.colab import drive
# 
# drive.mount('./gdrive')
# driveinfile = "./vid/test1.mp4"
# driveoutfile = './output/output.mp4'

driveinfile = "./vid/test_multi.mp4"
driveoutfile = './output/output_multi.mp4'

# VideoReaderWriter
cap = cv2.VideoCapture(driveinfile)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc('h','2','6', '4')
video = cv2.VideoWriter(driveoutfile, fourcc, fps, (width, height))

model = 101
###scale_factor = 1.0
scale_factor = 0.4

with tf.Session() as sess:
    print('MODEL-INIT:')
    ####model_cfg, model_outputs = posenet.load_model(args.model, sess)
    model_cfg, model_outputs = posenet.load_model(model, sess)
    output_stride = model_cfg['output_stride']
    start = time.time()
    print('START:')

    incnt = 0
    while True:
        incnt = incnt + 1
        try: input_image, draw_image, output_scale = posenet.read_cap(
                cap, scale_factor=scale_factor, output_stride=output_stride)
        except:break
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

        draw_image = posenet.draw_skel_and_kp(
                draw_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.25, min_part_score=0.25)
        
        cv2.imshow('posenet', draw_image)

        video.write(draw_image)

        if incnt % 100 == 0:        
            print("cnt=", incnt, "fps=", incnt / (time.time() - start) )

        if False:
            #debug
            for pi in range(len(pose_scores)):
                if pose_scores[pi] == 0.:
                    break
                print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                    print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))
video.release()
cap.release()
print('END:')
