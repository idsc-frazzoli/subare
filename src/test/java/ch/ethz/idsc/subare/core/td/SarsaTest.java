// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.adapter.SimpleTestModel;
import ch.ethz.idsc.subare.core.adapter.SimpleTestModels;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.subare.core.util.StateAction;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import junit.framework.TestCase;

public class SarsaTest extends TestCase {
  public void testConstantOneExact() {
    for (SarsaType sarsaType : SarsaType.values()) {
      MonteCarloInterface monteCarloInterface = SimpleTestModel.INSTANCE;
      LearningRate learningRate = ConstantLearningRate.one();
      DiscreteQsa qsa = DiscreteQsa.build(monteCarloInterface, RealScalar.ZERO);
      Sarsa sarsa = sarsaType.supply(monteCarloInterface, learningRate, qsa);
      assertFalse(sarsa.sac().isEncountered(StateAction.key(RealScalar.ZERO, RealScalar.ONE)));
      Scalar epsilon = RealScalar.of(.2);
      sarsa.setExplore(epsilon);
      Policy policy = EGreedyPolicy.bestEquiprobable(monteCarloInterface, qsa, epsilon);
      ExploringStarts.batch(monteCarloInterface, policy, 2, sarsa); // nstep > 1 required
      // DiscreteUtils.print(qsa);
      SimpleTestModels._checkExact(qsa);
      assertTrue(sarsa.sac().isEncountered(StateAction.key(RealScalar.ZERO, RealScalar.ONE)));
    }
  }

  public void testConstantNonOneExact() {
    for (SarsaType sarsaType : SarsaType.values()) {
      MonteCarloInterface monteCarloInterface = SimpleTestModel.INSTANCE;
      DiscreteQsa qsa = DiscreteQsa.build(monteCarloInterface, RealScalar.ZERO);
      LearningRate learningRate = ConstantLearningRate.of(RealScalar.of(.8));
      Sarsa sarsa = sarsaType.supply(monteCarloInterface, learningRate, qsa);
      assertFalse(sarsa.sac().isEncountered(StateAction.key(RealScalar.ZERO, RealScalar.ONE)));
      Scalar epsilon = RealScalar.of(.2);
      sarsa.setExplore(epsilon);
      Policy policy = EGreedyPolicy.bestEquiprobable(monteCarloInterface, qsa, epsilon);
      ExploringStarts.batch(monteCarloInterface, policy, 2, sarsa); // nstep > 1 required
      // DiscreteUtils.print(qsa);
      SimpleTestModels._checkExact(qsa);
      assertTrue(sarsa.sac().isEncountered(StateAction.key(RealScalar.ZERO, RealScalar.ONE)));
    }
  }

  public void testDefaultExact() {
    for (SarsaType sarsaType : SarsaType.values()) {
      MonteCarloInterface monteCarloInterface = SimpleTestModel.INSTANCE;
      LearningRate learningRate = DefaultLearningRate.of(1.5, 0.6);
      DiscreteQsa qsa = DiscreteQsa.build(monteCarloInterface, RealScalar.ZERO);
      Sarsa sarsa = sarsaType.supply(monteCarloInterface, learningRate, qsa);
      assertFalse(sarsa.sac().isEncountered(StateAction.key(RealScalar.ZERO, RealScalar.ONE)));
      Scalar epsilon = RealScalar.of(.2);
      sarsa.setExplore(epsilon);
      Policy policy = EGreedyPolicy.bestEquiprobable(monteCarloInterface, qsa, epsilon);
      ExploringStarts.batch(monteCarloInterface, policy, 2, sarsa); // nstep > 1 required
      // DiscreteUtils.print(qsa);
      SimpleTestModels._checkExact(qsa);
      assertTrue(sarsa.sac().isEncountered(StateAction.key(RealScalar.ZERO, RealScalar.ONE)));
    }
  }
}
