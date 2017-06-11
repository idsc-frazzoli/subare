// code by jph
package ch.ethz.idsc.subare.ch03.grid;

import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.alg.Random1StepTabularQPlanning;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.TabularSteps;
import ch.ethz.idsc.subare.core.util.TensorValuesUtils;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensors;
import junit.framework.TestCase;

public class GridWorldTest extends TestCase {
  public void testBasics() {
    Gridworld gw = new Gridworld();
    assertEquals(gw.reward(Tensors.vector(0, 0), Tensors.vector(1, 0), null), RealScalar.ZERO);
    assertEquals(gw.reward(Tensors.vector(0, 0), Tensors.vector(-1, 0), null), RealScalar.ONE.negate());
  }

  public void testIndex() {
    Gridworld gw = new Gridworld();
    Index actionsIndex = Index.build(gw.actions);
    int index = actionsIndex.of(Tensors.vector(1, 0));
    assertEquals(index, 3);
  }

  public void testR1STQL() {
    Gridworld gridworld = new Gridworld();
    DiscreteQsa ref = ActionValueIterations.getOptimal(gridworld, DecimalScalar.of(0.0001));
    DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    Random1StepTabularQPlanning rstqp = new Random1StepTabularQPlanning(gridworld, qsa);
    rstqp.setLearningRate(RealScalar.of(1.));
    for (int index = 0; index < 40; ++index)
      TabularSteps.batch(gridworld, gridworld, rstqp);
    assertTrue(Scalars.lessThan(TensorValuesUtils.distance(ref, qsa), RealScalar.of(2)));
  }
}
