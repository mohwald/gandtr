import unittest

from mdir.hub.model import gem_vgg16_cyclegan, gem_resnet101_cyclegan, gem_vgg16_hedngan, gem_resnet101_hedngan, \
    cyclegan, hedngan

entrypoints = [
    gem_vgg16_cyclegan,
    gem_resnet101_cyclegan,
    gem_vgg16_hedngan,
    gem_resnet101_hedngan,
    cyclegan,
    hedngan,
]


class TestHubInit(unittest.TestCase):
    def test_can_initialize_base_models(self):
        for p in entrypoints:
            with self.subTest(msg=p.__class__.__name__):
                self.assertIsNotNone(p(pretrained=False))

    def test_can_initialize_pretrained_models(self):
        for p in entrypoints:
            with self.subTest(msg=p.__class__.__name__):
                self.assertIsNotNone(p(pretrained=True))
