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
import ch.ethz.idsc.tensor.RealScalar;
import junit.framework.TestCase;

public class DoubleSarsaTest extends TestCase {
  public void testExact() {
    for (SarsaType sarsaType : SarsaType.values()) {
      MonteCarloInterface monteCarloInterface = SimpleTestModel.INSTANCE;
      LearningRate learningRate = DefaultLearningRate.of(RealScalar.of(2.3), RealScalar.of(0.6));
      DiscreteQsa qsa = DiscreteQsa.build(monteCarloInterface, RealScalar.ZERO);
      DiscreteQsa qsa1 = DiscreteQsa.build(monteCarloInterface, RealScalar.ZERO);
      DiscreteQsa qsa2 = DiscreteQsa.build(monteCarloInterface, RealScalar.ZERO);
      StateActionCounter sac = new DiscreteStateActionCounter();
      StateActionCounter sac1 = new DiscreteStateActionCounter();
      StateActionCounter sac2 = new DiscreteStateActionCounter();
      PolicyBase policy = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa, sac);
      PolicyBase policy1 = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa1, sac1);
      PolicyBase policy2 = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa2, sac2);
      DoubleSarsa doubleSarsa = sarsaType.doubleSarsa(monteCarloInterface, learningRate, qsa1, qsa2, sac1, sac2, policy1, policy2);
      ExploringStarts.batch(monteCarloInterface, policy, 2, doubleSarsa); // nstep > 1 required
      // DiscreteUtils.print(doubleSarsa.qsa());
      // TODO JAN investigate why this results in numeric precision
      SimpleTestModels._checkExactNumeric(doubleSarsa.qsa());
    }
  }

  public void testExact2() {
    for (SarsaType sarsaType : SarsaType.values()) {
      MonteCarloInterface monteCarloInterface = SimpleTestModel.INSTANCE;
      LearningRate learningRate = ConstantLearningRate.one();
      DiscreteQsa qsa = DiscreteQsa.build(monteCarloInterface, RealScalar.ZERO);
      DiscreteQsa qsa1 = DiscreteQsa.build(monteCarloInterface, RealScalar.ZERO);
      DiscreteQsa qsa2 = DiscreteQsa.build(monteCarloInterface, RealScalar.ZERO);
      StateActionCounter sac = new DiscreteStateActionCounter();
      StateActionCounter sac1 = new DiscreteStateActionCounter();
      StateActionCounter sac2 = new DiscreteStateActionCounter();
      PolicyBase policy = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa, sac);
      PolicyBase policy1 = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa1, sac1);
      PolicyBase policy2 = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa2, sac2);
      DoubleSarsa doubleSarsa = sarsaType.doubleSarsa(monteCarloInterface, learningRate, qsa1, qsa2, sac1, sac2, policy1, policy2);
      ExploringStarts.batch(monteCarloInterface, policy, 2, doubleSarsa); // nstep > 1 required
      // DiscreteUtils.print(doubleSarsa.qsa());
      SimpleTestModels._checkExact(doubleSarsa.qsa());
    }
  }
}
