// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.adapter.SimpleTestModel;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import junit.framework.TestCase;

public class DoubleSarsaTest extends TestCase {
  public void testExact() {
    for (SarsaType sarsaType : SarsaType.values()) {
      System.out.println("---");
      MonteCarloInterface monteCarloInterface = SimpleTestModel.INSTANCE;
      LearningRate learningRate1 = ConstantLearningRate.of(RationalScalar.HALF);
      LearningRate learningRate2 = ConstantLearningRate.of(RationalScalar.of(3, 4));
      DiscreteQsa qsa1 = DiscreteQsa.build(monteCarloInterface, RealScalar.ZERO);
      DiscreteQsa qsa2 = DiscreteQsa.build(monteCarloInterface, RealScalar.ZERO);
      DoubleSarsa doubleSarsa = sarsaType.doubleSarsa(monteCarloInterface, learningRate1, learningRate2, qsa1, qsa2);
      Scalar epsilon = RealScalar.of(.2);
      doubleSarsa.setExplore(epsilon);
      Policy policy = EGreedyPolicy.bestEquiprobable(monteCarloInterface, qsa1, epsilon);
      ExploringStarts.batch(monteCarloInterface, policy, 2, doubleSarsa); // nstep > 1 required
      doubleSarsa.qsa();
      DiscreteUtils.print(qsa1);
      DiscreteUtils.print(qsa2);
      // FIXME JAN/FLURIC
    }
  }
}
