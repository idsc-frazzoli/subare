// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.adapter.SimpleTestModel;
import ch.ethz.idsc.subare.core.adapter.SimpleTestModels;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import junit.framework.TestCase;

public class DoubleSarsaTest extends TestCase {
  public void testExact() {
    for (SarsaType sarsaType : SarsaType.values()) {
      MonteCarloInterface monteCarloInterface = SimpleTestModel.INSTANCE;
      LearningRate learningRate1 = DefaultLearningRate.of(RealScalar.of(2.3), RealScalar.of(0.6));
      LearningRate learningRate2 = DefaultLearningRate.of(RealScalar.of(2.3), RealScalar.of(0.6));
      DiscreteQsa qsa1 = DiscreteQsa.build(monteCarloInterface, RealScalar.ZERO);
      DiscreteQsa qsa2 = DiscreteQsa.build(monteCarloInterface, RealScalar.ZERO);
      DoubleSarsa doubleSarsa = sarsaType.doubleSarsa(monteCarloInterface, learningRate1, learningRate2, qsa1, qsa2);
      Scalar epsilon = RealScalar.of(.2);
      doubleSarsa.setExplore(epsilon);
      Policy policy = EGreedyPolicy.bestEquiprobable(monteCarloInterface, qsa1, epsilon);
      ExploringStarts.batch(monteCarloInterface, policy, 2, doubleSarsa); // nstep > 1 required
      // DiscreteUtils.print(doubleSarsa.qsa());
      // TODO JAN investigate why this results in numeric precision
      SimpleTestModels._checkExactNumeric(doubleSarsa.qsa());
    }
  }

  public void testExact2() {
    for (SarsaType sarsaType : SarsaType.values()) {
      MonteCarloInterface monteCarloInterface = SimpleTestModel.INSTANCE;
      LearningRate learningRate1 = ConstantLearningRate.one();
      LearningRate learningRate2 = ConstantLearningRate.one();
      DiscreteQsa qsa1 = DiscreteQsa.build(monteCarloInterface, RealScalar.ZERO);
      DiscreteQsa qsa2 = DiscreteQsa.build(monteCarloInterface, RealScalar.ZERO);
      DoubleSarsa doubleSarsa = sarsaType.doubleSarsa(monteCarloInterface, learningRate1, learningRate2, qsa1, qsa2);
      Scalar epsilon = RealScalar.of(.2);
      doubleSarsa.setExplore(epsilon);
      Policy policy = EGreedyPolicy.bestEquiprobable(monteCarloInterface, qsa1, epsilon);
      ExploringStarts.batch(monteCarloInterface, policy, 2, doubleSarsa); // nstep > 1 required
      // DiscreteUtils.print(doubleSarsa.qsa());
      SimpleTestModels._checkExact(doubleSarsa.qsa());
    }
  }
}
