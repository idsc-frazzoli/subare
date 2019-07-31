// code by jph
package ch.ethz.idsc.subare.ch03.grid;

import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.alg.Random1StepTabularQPlanning;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteValueFunctions;
import ch.ethz.idsc.subare.core.util.TabularSteps;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensors;
import junit.framework.TestCase;

public class GridworldTest extends TestCase {
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
    DiscreteQsa ref = ActionValueIterations.solve(gridworld, DecimalScalar.of(0.0001));
    DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    Random1StepTabularQPlanning rstqp = Random1StepTabularQPlanning.of( //
        gridworld, qsa, ConstantLearningRate.of(RealScalar.ONE));
    Scalar error = null;
    for (int index = 0; index < 40; ++index) {
      TabularSteps.batch(gridworld, gridworld, rstqp);
      error = DiscreteValueFunctions.distance(ref, qsa);
    }
    assertTrue(Scalars.lessThan(error, RealScalar.of(3)));
  }
}
