// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.adapter.SimpleTestModel;
import ch.ethz.idsc.subare.core.adapter.SimpleTestModels;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteStateActionCounter;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.subare.core.util.PolicyBase;
import ch.ethz.idsc.subare.core.util.PolicyType;
import ch.ethz.idsc.tensor.RealScalar;
import junit.framework.TestCase;

public class PrioritizedSweepingTest extends TestCase {
  public void testSimple() {
    SimpleTestModel simpleTestModel = SimpleTestModel.INSTANCE;
    LearningRate learningRate = DefaultLearningRate.of(8, 2);
    DiscreteQsa qsa = DiscreteQsa.build(simpleTestModel, RealScalar.ZERO);
    StateActionCounter sac = new DiscreteStateActionCounter();
    PolicyBase policy = PolicyType.EGREEDY.bestEquiprobable(simpleTestModel, qsa, sac);
    Sarsa sarsa = SarsaType.ORIGINAL.sarsa(simpleTestModel, learningRate, qsa, sac, policy);
    PrioritizedSweeping ps = new PrioritizedSweeping(sarsa, 2, RealScalar.of(.1));
    ExploringStarts.batch(simpleTestModel, policy, ps);
    SimpleTestModels._checkExact(qsa);
  }
}
