from unittest import TestCase

from common.utilities import Empty


class TestEmpty(TestCase):
    def setUp(self):
        self.sut = Empty()

    def test_set_field(self):
        value = 'doei'
        self.sut.hoi = value

        self.assertEqual(self.sut.hoi, value)