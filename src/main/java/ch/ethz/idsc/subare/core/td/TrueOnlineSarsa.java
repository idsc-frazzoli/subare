// code by fluric
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.TrueOnlineInterface;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.subare.core.util.FeatureQsaAdapter;
import ch.ethz.idsc.subare.core.util.FeatureWeight;
import ch.ethz.idsc.subare.core.util.StateAction;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.red.Times;
import ch.ethz.idsc.tensor.sca.Clip;

/** implementation of box "True Online Sarsa(lambda) for estimating w'x approx. q_pi or q_*
 * 
 * the evaluation types {@link Expected Reward} and {@link Q-Learning} are adapted from the original Sarsa
 * 
 * https://github.com/idsc-frazzoli/subare/files/2257720/trueOnlineSarsa.pdf
 * 
 * in Section 12.8, p.309 */
public class TrueOnlineSarsa implements TrueOnlineInterface {
  private final MonteCarloInterface monteCarloInterface;
  private final FeatureMapper featureMapper;
  private final LearningRate learningRate;
  // ---
  private final Scalar gamma;
  private final Scalar gamma_lambda;
  private final int featureSize;
  private final SarsaEvaluation evaluationType;
  /** feature weight vector w is a long-term memory, accumulating over the lifetime of the system */
  private final FeatureWeight w;
  // ---
  private Scalar epsilon;
  // ---
  private Scalar nextQOld;
  /** eligibility trace z is a short-term memory, typically lasting less time than the length of an episode */
  private Tensor z;

  /* package */ TrueOnlineSarsa( //
      MonteCarloInterface monteCarloInterface, SarsaEvaluation evaluationType, //
      Scalar lambda, FeatureMapper featureMapper, LearningRate learningRate, FeatureWeight w) {
    this.monteCarloInterface = monteCarloInterface;
    this.evaluationType = evaluationType;
    this.learningRate = learningRate;
    Clip.unit().requireInside(lambda);
    this.gamma = monteCarloInterface.gamma();
    this.featureMapper = featureMapper;
    gamma_lambda = Times.of(gamma, lambda);
    featureSize = featureMapper.featureSize();
    this.w = w;
    resetEligibility();
  }

  /** @param epsilon in [0, 1]
   * @throws Exception if input is outside valid range */
  public final void setExplore(Scalar epsilon) {
    this.epsilon = Clip.unit().requireInside(epsilon);
  }

  /** Returns the Qsa according to the current feature weights.
   * Only use this function, when the state-action space is small enough. */
  @Override // from DiscreteQsaSupplier
  public final DiscreteQsa qsa() {
    DiscreteQsa qsa = DiscreteQsa.build(monteCarloInterface);
    for (Tensor state : monteCarloInterface.states())
      for (Tensor action : monteCarloInterface.actions(state)) {
        Tensor stateActionPair = StateAction.key(state, action);
        qsa.assign(state, action, featureMapper.getFeature(stateActionPair).dot(w.get()).Get());
      }
    return qsa;
  }

  /** faster when only part of the qsa is required */
  @Override // from QsaInterfaceSupplier
  public final QsaInterface qsaInterface() {
    return new FeatureQsaAdapter(w.get(), featureMapper);
  }

  /** @return unmodifiable weight vector w */
  public final Tensor getW() {
    return w.get().unmodifiable();
  }

  @Override // from StepDigest
  public final void digest(StepInterface stepInterface) {
    Tensor prevState = stepInterface.prevState();
    Tensor prevAction = stepInterface.action();
    Tensor nextState = stepInterface.nextState();
    Scalar reward = stepInterface.reward();
    // ---
    Scalar alpha = learningRate.alpha(stepInterface);
    Scalar alpha_gamma_lambda = Times.of(alpha, gamma_lambda);
    Tensor x = featureMapper.getFeature(StateAction.key(prevState, prevAction));
    Scalar prevQ = w.get().dot(x).Get();
    Scalar nextQ = evaluationType.evaluate(epsilon, learningRate, nextState, qsaInterface());
    Scalar delta = reward.add(gamma.multiply(nextQ)).subtract(prevQ);
    // eq (12.11)
    z = z.multiply(gamma_lambda) //
        .add(x.multiply(RealScalar.ONE.subtract(alpha_gamma_lambda.multiply(z.dot(x).Get()))));
    // ---
    Scalar diffQ = prevQ.subtract(nextQOld);
    Tensor scalez = z.multiply(alpha.multiply(delta.add(diffQ)));
    Tensor scalex = x.multiply(alpha.multiply(diffQ));
    w.set(w.get().add(scalez).subtract(scalex));
    nextQOld = nextQ;
    // ---
    learningRate.digest(stepInterface);
    // ---
    if (monteCarloInterface.isTerminal(nextState))
      resetEligibility();
  }

  private final void resetEligibility() {
    nextQOld = RealScalar.ZERO;
    /** eligibility trace vector is initialized to zero at the beginning of the
     * episode */
    z = Array.zeros(featureSize);
  }
}
