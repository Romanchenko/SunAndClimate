from unittest import TestCase

import numpy
import matplotlib.pyplot as plt

import sunny


class Test(TestCase):

    def test_in_spring_delta(self):
        self.assertFalse(sunny.in_spring_delta(1, 5))
        self.assertTrue(sunny.in_spring_delta(79, 0))
        self.assertFalse(sunny.in_spring_delta(70, 5))

    def test_in_autumn_delta(self):
        self.assertFalse(sunny.in_autumn_delta(260, 2))
        self.assertTrue(sunny.in_autumn_delta(260, 10))
        self.assertTrue(sunny.in_autumn_delta(265, 1))


class TestGenerative(TestCase):

    def test1(self):
        def foo1(step, prev, gen):
            return prev * 2 + 10

        generator = sunny.Generative(foo1)
        self.assertEqual(-30, generator.next())
        self.assertEqual(-50, generator.next())
        generator.reset()
        self.assertEqual(-30, generator.next())

    def test2(self):
        def foo(step, prev, gen):
            return step * prev

        generator = sunny.Generative(foo)
        self.assertEqual(-20, generator.next())
        self.assertEqual(-40, generator.next())
        self.assertEqual(-120, generator.next())

    def test3(self):
        def foo(step, prev, gen):
            return gen.normal(3, 0.5)

        generator = sunny.Generative(foo)
        summ = 0
        n = 1000000
        for i in range(n):
            summ += generator.next()
        mean = summ / n
        self.assertAlmostEqual(abs(3.0 - mean), 0, delta=1e-3)


class Test(TestCase):
    def test_m_fft(self):
        freq = 2 * numpy.pi / 123
        n = 100000
        amp_expected = 13
        arr = [5 + numpy.sin(freq * i) * amp_expected * 2 for i in range(n)]
        amps, n1 = sunny.mFFT(arr, draw=False, return_pair=True)
        eps = 1e-2
        amp_actual = amps[(n1 - 123) < eps].max()
        self.assertAlmostEqual(amp_expected, amp_actual, delta=eps)

    def test_m_ffte(self):
        freq1 = 2 * numpy.pi / 27
        freq2 = 2 * numpy.pi / 183
        n = 108 * 183
        arr = [numpy.sin(freq1 * i) + numpy.sin(freq2 * i) for i in range(n)]
        amps, en1, en2, relations = sunny.mFFTe(arr, ep1=27, ep2=183, radius1=1, radius2=2)
        self.assertAlmostEqual(0.5, en1, delta=1e-2)
        self.assertAlmostEqual(0.5, en2, delta=1e-2)


class Test(TestCase):
    def test_imitate_dst_new(self):
        def foo(prev, step, gen):
            return step + prev
        generator = sunny.Generative(foo, initial_value=0)
        res = sunny.imitate_Dst_new(generator, 10)
        self.assertEqual(1, res[0])
        self.assertEqual(3, res[1])
        self.assertEqual(6, res[2])
        self.assertEqual(10, res[3])
        self.assertEqual(15, res[4])
