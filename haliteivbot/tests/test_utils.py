import unittest

from haliteivbot.utils import *


class TestUtils(unittest.TestCase):

    def test_navigation_map(self):
        """
        Test that the navigation map contains the right directions
        """
        create_navigation_lists(21)
        print(NAVIGATION)

        source1 = 10
        target1 = 11
        directions1 = nav(source1, target1)
        self.assertEqual(directions1, [ShipAction.EAST])

        source2 = 0
        target2 = 39
        directions2 = nav(source2, target2)
        self.assertEqual(directions2, [ShipAction.WEST, ShipAction.SOUTH])  # wrap-around


if __name__ == '__main__':
    unittest.main()
