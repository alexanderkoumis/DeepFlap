
# import cv2
import numpy as np
import skimage.transform

from flappy import flappy


def show_image(image):
    cv2.imshow('cool', image)
    cv2.waitKey(1)


class FlappyEnv(object):

    action_space = ['UP', 'IDLE']

    def __init__(self, reward, frames_in_state=4, image_scale=4):
        self.reward = reward
        self.frames_in_state = frames_in_state
        self.image_dims = (
            int(flappy.SCREENHEIGHT / image_scale),
            int(flappy.SCREENWIDTH / image_scale)
        )
        self.reset()

    def state_size(self):
        rows, cols = self.image_dims
        return (rows, cols, self.frames_in_state)

    def reset(self):
        self._step = flappy.main(self.reward)

    def peak(self):
        images, score, alive = self.step('IDLE')
        return images

    def step(self, action):

        # Pass 'IDLE' for first N-1 frames
        images = []
        score_total = 0

        for _ in range(self.frames_in_state-1):
            image, score_frame, alive = self._step('IDLE')
            images.append(self.process_image(image))
            score_total += score_frame

        # Pass action for last frame
        image, score_frame, alive = self._step(action)
        images.append(self.process_image(image))
        score_total += score_frame

        images = np.dstack(images)
        # show_image(self.process_image(image, True))

        if not alive:
            self.reset()
            score_total = 0

        return images, score_total, int(alive)

    def process_image(self, image, human_viewable=False):

        if human_viewable:
            # Rearrange rgb channels and flip
            image = np.transpose(image[...,::-1], (1, 0, 2))

        # Convert to grayscale
        image = image.dot([0.298, 0.587, 0.114])

        image_scaled = skimage.transform.resize(image, self.image_dims)

        return image_scaled.astype('uint8')


def main():
    env = FlappyEnv()
    env.reset()

    while True:

        images, score, alive = env.step('IDLE')
        if alive:
            env.reset()

if __name__ == '__main__':
    main()