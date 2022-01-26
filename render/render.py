import argparse
import os
import sys

import bpy
import numpy as np
from scipy.spatial.distance import squareform, pdist

def render(args):
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    bpy.ops.wm.open_mainfile(filepath=args.scene)

    trajs = np.load(args.trajs)[:, :, [0, 2, 1]] * np.array([1, -1, 1])

    for i, points in enumerate(trajs):
        print('==================================================')
        print(f'Rendering frame {i+1} of {trajs.shape[0]}.')
        print('==================================================')
        neighbors = squareform(pdist(points)).argsort()

        d = np.linalg.norm(points[neighbors[:, 11]] - points, axis=1)

        mball = bpy.data.metaballs.new("Inner")
        mball.resolution = 0.02
        mball.render_resolution = 0.02
        mball.threshold = 0.5
        mball_obj = bpy.data.objects.new("Inner", mball)
        bpy.context.view_layer.active_layer_collection.collection.objects.link(mball_obj)

        for r, co in zip(d, points):
            ele = mball.elements.new()
            ele.radius = r
            ele.co = co
            
        mball.materials.append(bpy.data.materials['MetaballInner'])

        mball_obj.select_set(True)
        bpy.context.view_layer.objects.active = mball_obj
        bpy.ops.object.convert(target="MESH")

        smooth = bpy.context.selected_objects[0].modifiers.new("Smooth", "SMOOTH")
        smooth.iterations = args.smooth_iters
        smooth.factor = args.smooth_factor
        bpy.context.selected_objects[0].select_set(False)

        d = np.linalg.norm(points[neighbors[:, 11]] - points, axis=1)

        mball = bpy.data.metaballs.new("Outer")
        mball.resolution = 0.02
        mball.render_resolution = 0.02
        mball.threshold = 0.5
        mball_obj = bpy.data.objects.new("Outer", mball)
        bpy.context.view_layer.active_layer_collection.collection.objects.link(mball_obj)

        for r, co in zip(d, points):
            ele = mball.elements.new()
            ele.radius = r * 1.5
            ele.co = co
            
        mball.materials.append(bpy.data.materials['MetaballOuter'])

        mball_obj.select_set(True)
        bpy.context.view_layer.objects.active = mball_obj
        bpy.ops.object.convert(target="MESH")

        smooth = bpy.context.selected_objects[0].modifiers.new("Smooth", "SMOOTH")
        smooth.iterations = args.smooth_iters
        smooth.factor = args.smooth_factor
        bpy.context.selected_objects[0].select_set(False)

        bpy.context.scene.render.filepath = os.path.join(args.out,
                                                         f'frame{i}.png')
        bpy.ops.render.render(write_still=1)

        bpy.data.objects.remove(bpy.data.objects['Inner.001'])
        bpy.data.objects.remove(bpy.data.objects['Outer.001'])

    print('==================================================')
    print('Generating video.')
    print('==================================================')
    os.system(f'ffmpeg -y -i {os.path.join(args.out, "frame%d.png")} {os.path.join(args.out, "video.mp4")}')


if '--' in sys.argv:
    argv = sys.argv[sys.argv.index('--') + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajs', type=str, default='trajs.npy')
    parser.add_argument('--scene', type=str, default='scene.blend')
    parser.add_argument('--out', type=str, default='out')
    parser.add_argument('--smooth_factor', type=int, default=0.5)
    parser.add_argument('--smooth_iters', type=int, default=20)
    args = parser.parse_known_args(argv)[0]
    render(args)
