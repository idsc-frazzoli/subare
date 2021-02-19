// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.subare.util.AssertFail;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.num.Series;
import junit.framework.TestCase;

public class DiscountFunctionTest extends TestCase {
  public void testSimple() {
    Tensor coeffs = Tensors.vector(3, 2, -3, 2, .3);
    DiscountFunction discountFunction = DiscountFunction.of(RealScalar.ONE);
    Scalar gain1 = discountFunction.apply(coeffs);
    Scalar gain2 = Series.of(coeffs).apply(RealScalar.ONE);
    assertEquals(gain1, gain2);
  }

  public void testHorner() {
    Tensor coeffs = Tensors.vector(3, 2, -3, 2, .3);
    Scalar alpha = RealScalar.of(.2);
    DiscountFunction discountFunction = DiscountFunction.of(alpha);
    Scalar gain1 = discountFunction.apply(coeffs);
    Scalar gain2 = Series.of(coeffs).apply(alpha);
    assertEquals(gain1, gain2);
  }

  public void testFail() {
    AssertFail.of(() -> DiscountFunction.of(RealScalar.of(1.1)));
  }
}
