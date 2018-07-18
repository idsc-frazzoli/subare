// code by fluric
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.Max;

public class QLearningTrueOnlineSarsa extends TrueOnlineSarsa {
  public static TrueOnlineSarsa of( //
      MonteCarloInterface monteCarloInterface, Scalar lambda, LearningRate learningRate, FeatureMapper featureMapper) {
    return new QLearningTrueOnlineSarsa(monteCarloInterface, lambda, learningRate, featureMapper, null);
  }

  public static TrueOnlineSarsa of( //
      MonteCarloInterface monteCarloInterface, Scalar lambda, LearningRate learningRate, FeatureMapper featureMapper, Tensor w) {
    return new QLearningTrueOnlineSarsa(monteCarloInterface, lambda, learningRate, featureMapper, w);
  }

  // ---
  private QLearningTrueOnlineSarsa(MonteCarloInterface monteCarloInterface, Scalar lambda, LearningRate learningRate, FeatureMapper featureMapper, Tensor w) {
    super(monteCarloInterface, lambda, learningRate, featureMapper, w);
  }

  @Override // from TrueOnlineSarsa
  protected Scalar evaluate(StepInterface stepInterface) {
    Tensor nextState = stepInterface.nextState();
    QsaInterface qsaInterface = qsaInterface();
    return monteCarloInterface.actions(nextState).stream() //
        .filter(action -> learningRate.encountered(nextState, action)) //
        .map(action -> qsaInterface.value(nextState, action)) //
        .reduce(Max::of) //
        .orElse(RealScalar.ZERO);
  }
}
