
# import cv2
import numpy as np
import skimage.transform

from flappy import flappy


def show_image(image):
    cv2.imshow('cool', image)
    cv2.waitKey(1)


class FlappyEnv(object):

    action_space = ['UP', 'IDLE']

    def __init__(self, frames_in_state=4, image_scale=5):
        self.frames_in_state = frames_in_state
        self.image_dims = (
            int(flappy.SCREENHEIGHT / image_scale),
            int(flappy.SCREENWIDTH / image_scale)
        )

    def state_size(self):
        rows, cols = self.image_dims
        return (rows, cols, self.frames_in_state)

    def reset(self):
        self._step = flappy.main()

    def peak(self):
        images, score, dead = self.step('IDLE')
        return images

    def step(self, action):

        # Pass 'IDLE' for first N-1 frames
        images = []
        for _ in range(self.frames_in_state-1):
            image, score, dead = self._step('IDLE')
            images.append(self.process_image(image))

        # Pass action for last frame
        image, score, dead = self._step(action)
        images.append(self.process_image(image))

        images = np.dstack(images)
        # show_image(self.process_image(image, True))

        return images, score, dead

    def process_image(self, image, human_viewable=False):

        if human_viewable:
            # Rearrange rgb channels and flip
            image = np.transpose(image[...,::-1], (1, 0, 2))

        # Convert to grayscale
        image = image.dot([0.298, 0.587, 0.114]).astype('uint8')

        return skimage.transform.resize(image, self.image_dims)


def main():
    env = FlappyEnv()
    env.reset()

    while True:

        images, score, dead = env.step('IDLE')
        if dead:
            env.reset()

if __name__ == '__main__':
    main()