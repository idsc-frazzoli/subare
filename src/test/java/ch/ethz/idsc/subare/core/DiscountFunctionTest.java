// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Multinomial;
import junit.framework.TestCase;

public class DiscountFunctionTest extends TestCase {
  public void testSimple() {
    Tensor coeffs = Tensors.vector(3, 2, -3, 2, .3);
    DiscountFunction discountFunction = DiscountFunction.of(RealScalar.ONE);
    Scalar gain1 = discountFunction.apply(coeffs);
    Scalar gain2 = Multinomial.horner(coeffs, RealScalar.ONE);
    assertEquals(gain1, gain2);
  }

  public void testHorner() {
    Tensor coeffs = Tensors.vector(3, 2, -3, 2, .3);
    Scalar alpha = RealScalar.of(.2);
    DiscountFunction discountFunction = DiscountFunction.of(alpha);
    Scalar gain1 = discountFunction.apply(coeffs);
    Scalar gain2 = Multinomial.horner(coeffs, alpha);
    assertEquals(gain1, gain2);
  }
}
