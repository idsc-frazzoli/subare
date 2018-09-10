// code by fluric
package ch.ethz.idsc.subare.analysis;

import java.util.ArrayDeque;
import java.util.Deque;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.subare.core.util.UcbPolicy;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.sca.Sign;

public class SarsaMonteCarloTrialUcb implements MonteCarloTrial {
  private static final Scalar ALPHA = RealScalar.of(0.05);
  private static final Scalar EPSILON = RealScalar.of(0.0);

  public static SarsaMonteCarloTrialUcb of(MonteCarloInterface monteCarloInterface, SarsaType sarsaType, LearningRate learningRate, DiscreteQsa qsa,
      int digestDepth) {
    return new SarsaMonteCarloTrialUcb(monteCarloInterface, sarsaType, learningRate, qsa, digestDepth);
  }

  public static SarsaMonteCarloTrialUcb of(MonteCarloInterface monteCarloInterface, SarsaType sarsaType) {
    return new SarsaMonteCarloTrialUcb(monteCarloInterface, sarsaType, ConstantLearningRate.of(ALPHA), DiscreteQsa.build(monteCarloInterface), 1);
  }

  // ---
  private final MonteCarloInterface monteCarloInterface;
  private final Sarsa sarsa;
  private final Deque<StepInterface> deque = new ArrayDeque<>();
  private final int digestDepth; // 0 is equal to the MonteCarlo approach

  private SarsaMonteCarloTrialUcb(MonteCarloInterface monteCarloInterface, SarsaType sarsaType, LearningRate learningRate, DiscreteQsa qsa, int digestDepth) {
    this.monteCarloInterface = monteCarloInterface;
    sarsa = sarsaType.supply(monteCarloInterface, learningRate, qsa);
    sarsa.setExplore(EPSILON);
    this.digestDepth = digestDepth;
    Sign.requirePositiveOrZero(RealScalar.of(digestDepth));
  }

  @Override // from MonteCarloTrial
  public void executeBatch() {
    Policy policy = UcbPolicy.bestEquiprobable(monteCarloInterface, sarsa.qsa(), sarsa.sac(), EPSILON);
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
