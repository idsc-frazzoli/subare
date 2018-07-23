// code by fluric
package ch.ethz.idsc.subare.analysis;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.td.DoubleSarsa;
import ch.ethz.idsc.subare.core.td.SarsaEvaluationType;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;

public class DoubleSarsaMonteCarloTrial implements MonteCarloTrial {
  private static final Scalar ALPHA = RealScalar.of(0.05);
  private static final Scalar EPSILON = RealScalar.of(0.1);
  private static final int DIGEST_DEPTH = 1; // 0 is equal to the MonteCarlo approach
  // ---
  private final MonteCarloInterface monteCarloInterface;
  private final DoubleSarsa doubleSarsa;

  public DoubleSarsaMonteCarloTrial(MonteCarloInterface monteCarloInterface, SarsaEvaluationType evaluationType) {
    this.monteCarloInterface = monteCarloInterface;
    DiscreteQsa qsa1 = DiscreteQsa.build(monteCarloInterface);
    DiscreteQsa qsa2 = DiscreteQsa.build(monteCarloInterface);
    LearningRate learningRate1 = ConstantLearningRate.of(ALPHA);
    LearningRate learningRate2 = ConstantLearningRate.of(ALPHA);
    doubleSarsa = new DoubleSarsa(evaluationType, monteCarloInterface, //
        qsa1, qsa2, //
        learningRate1, learningRate2);
    doubleSarsa.setExplore(EPSILON);
  }

  public void executeBatch() {
    Policy policy = doubleSarsa.getEGreedy();
    ExploringStarts.batch(monteCarloInterface, policy, DIGEST_DEPTH, doubleSarsa);
  }

  @Override
  public DiscreteQsa qsa() {
    return doubleSarsa.qsa();
  }
}
