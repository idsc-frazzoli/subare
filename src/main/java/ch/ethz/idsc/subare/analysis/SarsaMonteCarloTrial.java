// code by fluric
package ch.ethz.idsc.subare.analysis;

import java.util.ArrayDeque;
import java.util.Deque;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteStateActionCounter;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.subare.core.util.PolicyBase;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.sca.Sign;

public class SarsaMonteCarloTrial implements MonteCarloTrial {
  private static final Scalar ALPHA = RealScalar.of(0.05);

  public static SarsaMonteCarloTrial of(MonteCarloInterface monteCarloInterface, SarsaType sarsaType, LearningRate learningRate, DiscreteQsa qsa,
      StateActionCounter sac, PolicyBase policy, int digestDepth) {
    return new SarsaMonteCarloTrial(monteCarloInterface, sarsaType, learningRate, qsa, sac, policy, digestDepth);
  }

  public static SarsaMonteCarloTrial of(MonteCarloInterface monteCarloInterface, SarsaType sarsaType, PolicyBase policy) {
    return new SarsaMonteCarloTrial(monteCarloInterface, sarsaType, ConstantLearningRate.of(ALPHA), DiscreteQsa.build(monteCarloInterface),
        new DiscreteStateActionCounter(), policy, 1);
  }

  // ---
  private final MonteCarloInterface monteCarloInterface;
  private final Sarsa sarsa;
  private final Deque<StepInterface> deque = new ArrayDeque<>();
  private final PolicyBase policy;
  private final int digestDepth; // 0 is equal to the MonteCarlo approach

  private SarsaMonteCarloTrial(MonteCarloInterface monteCarloInterface, SarsaType sarsaType, LearningRate learningRate, DiscreteQsa qsa, StateActionCounter sac,
      PolicyBase policy, int digestDepth) {
    this.monteCarloInterface = monteCarloInterface;
    sarsa = sarsaType.supply(monteCarloInterface, learningRate, qsa, sac, policy);
    this.policy = policy;
    this.digestDepth = digestDepth;
    Sign.requirePositiveOrZero(RealScalar.of(digestDepth));
  }

  @Override // from MonteCarloTrial
  public void executeBatch() {
    ExploringStarts.batch(monteCarloInterface, policy, digestDepth, sarsa);
  }

  @Override // from MonteCarloTrial
  public DiscreteQsa qsa() {
    return sarsa.qsa();
  }

  @Override // from MonteCarloTrial
  public void digest(StepInterface stepInterface) {
    deque.add(stepInterface);
    if (!monteCarloInterface.isTerminal(stepInterface.nextState())) {
      if (deque.size() == digestDepth) { // never true, if nstep == 0
        sarsa.digest(deque);
        deque.poll();
      }
    } else {
      while (!deque.isEmpty()) {
        sarsa.digest(deque);
        deque.poll();
      }
    }
  }

  public int getDequeueSize() {
    return deque.size();
  }

  @Override // from MonteCarloTrial
  public QsaInterface qsaInterface() {
    return qsa();
  }
}
