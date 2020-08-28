// code by jph
package ch.ethz.idsc.subare.core.adapter;

import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.tensor.ExactScalarQ;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.sca.Chop;
import junit.framework.TestCase;

public class SimpleTestModels extends TestCase {
  public static void _checkExact(DiscreteQsa qsa) {
    Scalar value_s0_a1 = qsa.value(RealScalar.ZERO, RealScalar.ONE);
    assertEquals(value_s0_a1, RealScalar.of(3));
    assertTrue(ExactScalarQ.of(value_s0_a1));
    Scalar value_s1_a1 = qsa.value(RealScalar.ONE, RealScalar.ONE);
    assertEquals(value_s1_a1, RealScalar.ONE);
    assertTrue(ExactScalarQ.of(value_s1_a1));
    Scalar value_s2_a0 = qsa.value(RealScalar.of(2), RealScalar.ZERO);
    assertEquals(value_s2_a0, RealScalar.ZERO);
    assertTrue(ExactScalarQ.of(value_s2_a0));
  }

  public static void _checkExactNumeric(DiscreteQsa qsa) {
    Scalar value_s0_a1 = qsa.value(RealScalar.ZERO, RealScalar.ONE);
    assertEquals(value_s0_a1, RealScalar.of(3));
    Scalar value_s1_a1 = qsa.value(RealScalar.ONE, RealScalar.ONE);
    assertEquals(value_s1_a1, RealScalar.ONE);
    Scalar value_s2_a0 = qsa.value(RealScalar.of(2), RealScalar.ZERO);
    assertEquals(value_s2_a0, RealScalar.ZERO);
  }

  public static void _checkClose(DiscreteQsa qsa) {
    Scalar value_s0_a1 = qsa.value(RealScalar.ZERO, RealScalar.ONE);
    Scalar value_s1_a1 = qsa.value(RealScalar.ONE, RealScalar.ONE);
    Scalar value_s2_a0 = qsa.value(RealScalar.of(2), RealScalar.ZERO);
    assertEquals(value_s1_a1, RealScalar.ONE);
    assertEquals(value_s2_a0, RealScalar.ZERO);
    Chop CHOP_3 = Chop._03;
    if (!CHOP_3.isClose(value_s0_a1, RealScalar.of(3))) {
      DiscreteUtils.print(qsa);
      assertTrue(false);
    }
  }

  public void testAlibi() {
    // otherwise test case complains that no tests are implemented
  }
}
