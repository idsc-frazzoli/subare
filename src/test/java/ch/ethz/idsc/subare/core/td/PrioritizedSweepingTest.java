// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.adapter.SimpleTestModel;
import ch.ethz.idsc.subare.core.adapter.SimpleTestModels;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.tensor.RealScalar;
import junit.framework.TestCase;

public class PrioritizedSweepingTest extends TestCase {
  public void testSimple() {
    SimpleTestModel simpleTestModel = SimpleTestModel.INSTANCE;
    LearningRate learningRate = DefaultLearningRate.of(8, 2);
    DiscreteQsa qsa = DiscreteQsa.build(simpleTestModel, RealScalar.ZERO);
    Sarsa sarsa = SarsaType.ORIGINAL.supply(simpleTestModel, learningRate, qsa);
    PrioritizedSweeping ps = new PrioritizedSweeping(sarsa, 2, RealScalar.of(.1));
    Policy policy = EGreedyPolicy.bestEquiprobable(simpleTestModel, qsa, RealScalar.of(.1));
    ExploringStarts.batch(simpleTestModel, policy, ps);
    // DiscreteUtils.print(qsa);
    SimpleTestModels._checkExact(qsa);
  }
}
