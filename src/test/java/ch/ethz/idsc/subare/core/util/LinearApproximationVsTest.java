// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.VsInterface;
import ch.ethz.idsc.subare.core.adapter.CosineBasis;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.opt.TensorUnaryOperator;
import ch.ethz.idsc.tensor.sca.Chop;
import ch.ethz.idsc.tensor.sca.Clips;
import junit.framework.TestCase;

public class LinearApproximationVsTest extends TestCase {
  public void testSimple() {
    TensorUnaryOperator represent = CosineBasis.create(5, Clips.interval(0, 20));
    VsInterface vs = LinearApproximationVs.create(represent, Tensors.vector(0, 1, 0, 0, 0));
    Scalar value = vs.value(RealScalar.of(10));
    assertTrue(Chop._13.allZero(value));
  }
}
