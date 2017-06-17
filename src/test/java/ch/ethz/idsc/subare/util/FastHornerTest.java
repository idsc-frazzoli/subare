// code by jph
package ch.ethz.idsc.subare.util;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Multinomial;
import junit.framework.TestCase;

public class FastHornerTest extends TestCase {
  public void testSimple() {
    Tensor coeffs = Tensors.vector(3, 2, -3, 2, .3);
    Scalar gain1 = FastHorner.of(coeffs, RealScalar.ONE);
    Scalar gain2 = Multinomial.horner(coeffs, RealScalar.ONE);
    assertEquals(gain1, gain2);
  }

  public void testHorner() {
    Tensor coeffs = Tensors.vector(3, 2, -3, 2, .3);
    Scalar alpha = RealScalar.of(.2);
    Scalar gain1 = FastHorner.of(coeffs, alpha);
    Scalar gain2 = Multinomial.horner(coeffs, alpha);
    assertEquals(gain1, gain2);
  }
}
