// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.adapter.SimpleTestModel;
import ch.ethz.idsc.subare.core.adapter.SimpleTestModels;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteStateActionCounter;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.subare.core.util.PolicyBase;
import ch.ethz.idsc.subare.core.util.PolicyType;
import ch.ethz.idsc.subare.core.util.StateAction;
import ch.ethz.idsc.tensor.RealScalar;
import junit.framework.TestCase;

public class SarsaTest extends TestCase {
  public void testConstantOneExact() {
    for (SarsaType sarsaType : SarsaType.values()) {
      MonteCarloInterface monteCarloInterface = SimpleTestModel.INSTANCE;
      LearningRate learningRate = ConstantLearningRate.one();
      DiscreteQsa qsa = DiscreteQsa.build(monteCarloInterface);
      StateActionCounter sac = new DiscreteStateActionCounter();
      PolicyBase policy = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa, sac);
      Sarsa sarsa = sarsaType.sarsa(monteCarloInterface, learningRate, qsa, sac, policy);
      assertFalse(sarsa.sac().isEncountered(StateAction.key(RealScalar.ZERO, RealScalar.ONE)));
      ExploringStarts.batch(monteCarloInterface, policy, 2, sarsa); // nstep > 1 required
      // DiscreteUtils.print(qsa);
      SimpleTestModels._checkExact(qsa);
      assertTrue(sarsa.sac().isEncountered(StateAction.key(RealScalar.ZERO, RealScalar.ONE)));
    }
  }

  public void testConstantNonOneExact() {
    for (SarsaType sarsaType : SarsaType.values()) {
      MonteCarloInterface monteCarloInterface = SimpleTestModel.INSTANCE;
      DiscreteQsa qsa = DiscreteQsa.build(monteCarloInterface);
      LearningRate learningRate = ConstantLearningRate.of(RealScalar.of(.8));
      StateActionCounter sac = new DiscreteStateActionCounter();
      PolicyBase policy = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa, sac);
      Sarsa sarsa = sarsaType.sarsa(monteCarloInterface, learningRate, qsa, sac, policy);
      assertFalse(sarsa.sac().isEncountered(StateAction.key(RealScalar.ZERO, RealScalar.ONE)));
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
      DiscreteQsa qsa = DiscreteQsa.build(monteCarloInterface);
      StateActionCounter sac = new DiscreteStateActionCounter();
      PolicyBase policy = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa, sac);
      Sarsa sarsa = sarsaType.sarsa(monteCarloInterface, learningRate, qsa, sac, policy);
      assertFalse(sarsa.sac().isEncountered(StateAction.key(RealScalar.ZERO, RealScalar.ONE)));
      ExploringStarts.batch(monteCarloInterface, policy, 2, sarsa); // nstep > 1 required
      // DiscreteUtils.print(qsa);
      SimpleTestModels._checkExact(qsa);
      assertTrue(sarsa.sac().isEncountered(StateAction.key(RealScalar.ZERO, RealScalar.ONE)));
    }
  }
}
