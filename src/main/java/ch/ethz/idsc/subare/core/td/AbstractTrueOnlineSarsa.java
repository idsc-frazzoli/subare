// code by jph, fluric
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.StateActionCounterSupplier;
import ch.ethz.idsc.subare.core.TrueOnlineInterface;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.subare.core.util.StateAction;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.Times;
import ch.ethz.idsc.tensor.sca.Clips;

abstract class AbstractTrueOnlineSarsa implements TrueOnlineInterface, StateActionCounterSupplier {
  final MonteCarloInterface monteCarloInterface;
  final FeatureMapper featureMapper;
  final LearningRate learningRate;
  final SarsaEvaluation evaluationType;
  final Scalar gamma;
  final Scalar gamma_lambda;

  public AbstractTrueOnlineSarsa(MonteCarloInterface monteCarloInterface, SarsaEvaluation evaluationType, //
      Scalar lambda, LearningRate learningRate, //
      FeatureMapper featureMapper) {
    this.monteCarloInterface = monteCarloInterface;
    this.evaluationType = evaluationType;
    this.gamma = monteCarloInterface.gamma();
    gamma_lambda = Times.of(gamma, Clips.unit().requireInside(lambda));
    this.learningRate = learningRate;
    this.featureMapper = featureMapper;
  }

  /** Returns the Qsa according to the current feature weights.
   * Only use this function, when the state-action space is small enough. */
  @Override // from DiscreteQsaSupplier
  public final DiscreteQsa qsa() {
    DiscreteQsa qsa = DiscreteQsa.build(monteCarloInterface);
    for (Tensor state : monteCarloInterface.states())
      for (Tensor action : monteCarloInterface.actions(state)) {
        Tensor stateActionPair = StateAction.key(state, action);
        qsa.assign(state, action, featureMapper.getFeature(stateActionPair).dot(getW()).Get());
      }
    return qsa;
  }

  public abstract Tensor getW();
}
