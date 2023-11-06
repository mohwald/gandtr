import unittest

from mdir.hub.model import gem_vgg16_cyclegan, gem_resnet101_cyclegan, gem_vgg16_hedngan, gem_resnet101_hedngan, \
    generator_cyclegan, generator_hedngan

entrypoints = [
    gem_vgg16_cyclegan,
    gem_resnet101_cyclegan,
    gem_vgg16_hedngan,
    gem_resnet101_hedngan,
    generator_cyclegan,
    generator_hedngan,
]


class TestHubInit(unittest.TestCase):
    def test_can_initialize_models(self):
        for p in entrypoints:
            with self.subTest():
                self.assertIsNotNone(p())
