// code by jph
package ch.ethz.idsc.subare.ch05.infvar;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.alg.ValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import junit.framework.TestCase;

public class InfiniteVarianceTest extends TestCase {
  public void testActionValueIteration() {
    InfiniteVariance infiniteVariance = new InfiniteVariance();
    ActionValueIteration avi = ActionValueIteration.of(infiniteVariance);
    avi.untilBelow(RealScalar.of(.00001));
    DiscreteQsa qsa = avi.qsa();
    Scalar diff = qsa.value(InfiniteVariance.BACK, InfiniteVariance.BACK).subtract(RealScalar.ONE);
    assertTrue(Scalars.lessThan(diff.abs(), RealScalar.of(.001)));
    assertEquals(qsa.value(InfiniteVariance.BACK, InfiniteVariance.END), RealScalar.ZERO);
    assertEquals(qsa.value(InfiniteVariance.END, InfiniteVariance.END), RealScalar.ZERO);
  }

  public void testValueIteration() {
    InfiniteVariance infiniteVariance = new InfiniteVariance();
    ValueIteration vi = new ValueIteration(infiniteVariance);
    vi.untilBelow(RealScalar.of(.00001));
    DiscreteVs vs = vi.vs();
    Scalar diff = vs.value(InfiniteVariance.BACK).subtract(RealScalar.ONE);
    assertTrue(Scalars.lessThan(diff.abs(), RealScalar.of(.001)));
    assertEquals(vs.value(InfiniteVariance.END), RealScalar.ZERO);
  }
}
