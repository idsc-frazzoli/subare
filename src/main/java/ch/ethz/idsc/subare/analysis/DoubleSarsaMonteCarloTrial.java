// code by fluric
package ch.ethz.idsc.subare.analysis;

import java.util.ArrayDeque;
import java.util.Deque;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.td.DoubleSarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteStateActionCounter;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.subare.core.util.PolicyBase;
import ch.ethz.idsc.subare.core.util.PolicyType;
import ch.ethz.idsc.tensor.RealScalar;

/* package */ class DoubleSarsaMonteCarloTrial implements MonteCarloTrial {
  public static DoubleSarsaMonteCarloTrial of(MonteCarloInterface monteCarloInterface, SarsaType sarsaType) {
    DiscreteQsa qsa1 = DiscreteQsa.build(monteCarloInterface);
    DiscreteQsa qsa2 = DiscreteQsa.build(monteCarloInterface);
    StateActionCounter sac1 = new DiscreteStateActionCounter();
    StateActionCounter sac2 = new DiscreteStateActionCounter();
    PolicyBase policy1 = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa1, sac1);
    PolicyBase policy2 = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa2, sac2);
    return new DoubleSarsaMonteCarloTrial(monteCarloInterface, sarsaType, //
        ConstantLearningRate.of(RealScalar.of(0.05)), qsa1, qsa2, sac1, sac2, policy1, policy2);
  }

  private final static int DIGEST_DEPTH = 1;
  // ---
  private final MonteCarloInterface monteCarloInterface;
  private final DoubleSarsa doubleSarsa;
  private final Deque<StepInterface> deque = new ArrayDeque<>();

  public DoubleSarsaMonteCarloTrial( //
      MonteCarloInterface monteCarloInterface, SarsaType sarsaType, //
      LearningRate learningRate, //
      DiscreteQsa qsa1, DiscreteQsa qsa2, //
      StateActionCounter sac1, StateActionCounter sac2, //
      PolicyBase policy1, PolicyBase policy2) {
    this.monteCarloInterface = monteCarloInterface;
    doubleSarsa = sarsaType.doubleSarsa(monteCarloInterface, //
        learningRate, qsa1, qsa2, sac1, sac2, policy1, policy2);
  }

  @Override // from MonteCarloTrial
  public void executeBatch() {
    Policy policy = doubleSarsa.getPolicy();
    ExploringStarts.batch(monteCarloInterface, policy, DIGEST_DEPTH, doubleSarsa);
  }

  @Override // from MonteCarloTrial
  public DiscreteQsa qsa() {
    return doubleSarsa.qsa();
  }

  @Override // from MonteCarloTrial
  public void digest(StepInterface stepInterface) {
    deque.add(stepInterface);
    if (!monteCarloInterface.isTerminal(stepInterface.nextState())) {
      if (deque.size() == DIGEST_DEPTH) { // never true, if nstep == 0
        doubleSarsa.digest(deque);
        deque.poll();
      }
    } else {
      while (!deque.isEmpty()) {
        doubleSarsa.digest(deque);
        deque.poll();
      }
    }
  }

  @Override // from MonteCarloTrial
  public QsaInterface qsaInterface() {
    return qsa();
  }
}
