// code by jph
package ch.ethz.idsc.subare.core.adapter;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.opt.TensorUnaryOperator;
import ch.ethz.idsc.tensor.sca.Clip;
import junit.framework.TestCase;

public class CosineBasisTest extends TestCase {
  public void testLo() {
    TensorUnaryOperator fb = CosineBasis.create(4, Clip.function(50, 100));
    Tensor result = fb.apply(RealScalar.of(50));
    assertEquals(result, Tensors.vector(1, 1, 1, 1));
  }

  public void testHi() {
    TensorUnaryOperator fb = CosineBasis.create(4, Clip.function(0, 100));
    Tensor result = fb.apply(RealScalar.of(100));
    assertEquals(result, Tensors.vector(1, -1, 1, -1));
  }

  public void testFail() {
    TensorUnaryOperator tuo = CosineBasis.create(4, Clip.function(50, 100));
    try {
      tuo.apply(RealScalar.ZERO);
      assertTrue(false);
    } catch (Exception exception) {
      // ---
    }
  }
}
