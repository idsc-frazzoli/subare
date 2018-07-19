// code by fluric
package ch.ethz.idsc.subare.core.td;

import java.util.Objects;

import ch.ethz.idsc.subare.core.DiscreteQsaSupplier;
import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.subare.core.util.FeatureQsaAdapter;
import ch.ethz.idsc.subare.core.util.StateAction;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.red.Times;
import ch.ethz.idsc.tensor.sca.Clip;

/** implementation of box "True Online Sarsa(lambda) for estimating w'x approx. q_pi or q_*
 * 
 * in Section 12.8, p.309 */
public abstract class TrueOnlineSarsa implements DiscreteQsaSupplier, StepDigest {
  protected final MonteCarloInterface monteCarloInterface;
  protected final FeatureMapper featureMapper;
  protected final LearningRate learningRate;
  // ---
  private final Scalar gamma;
  private final Scalar gamma_lambda;
  private final int featureSize;
  // ---
  protected Scalar epsilon;
  /** weight vector w is a long-term memory, accumulating over the lifetime of the system */
  protected Tensor w;
  // ---
  private Scalar nextQOld;
  /** eligibility trace z is a short-term memory, typically lasting less time than the length of an episode */
  private Tensor z;

  /** @param monteCarloInterface
   * @param lambda in [0, 1] Figure 12.14 in the book suggests that lambda in [0.8, 0.9]
   * tends to be a good choice
   * @param learningRate
   * @param featureMapper
   * @param w */
  protected TrueOnlineSarsa(MonteCarloInterface monteCarloInterface, Scalar lambda, LearningRate learningRate, FeatureMapper featureMapper, Tensor w) {
    this.monteCarloInterface = monteCarloInterface;
    this.learningRate = learningRate;
    Clip.unit().requireInside(lambda);
    this.gamma = monteCarloInterface.gamma();
    this.featureMapper = featureMapper;
    gamma_lambda = Times.of(gamma, lambda);
    featureSize = featureMapper.featureSize();
    this.w = Objects.isNull(w) ? Array.zeros(featureSize) : w;
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
  public DiscreteQsa qsa() {
    DiscreteQsa qsa = DiscreteQsa.build(monteCarloInterface);
    for (Tensor state : monteCarloInterface.states())
      for (Tensor action : monteCarloInterface.actions(state)) {
        Tensor stateActionPair = StateAction.key(state, action);
        qsa.assign(state, action, featureMapper.getFeature(stateActionPair).dot(w).Get());
      }
    return qsa;
  }

  /** faster when only part of the qsa is required */
  public QsaInterface qsaInterface() {
    return new FeatureQsaAdapter(w, featureMapper);
  }

  /** @return unmodifiable weight vector w */
  public Tensor getW() {
    return w.unmodifiable();
  }

  @Override // from StepDigest
  public void digest(StepInterface stepInterface) {
    Tensor prevState = stepInterface.prevState();
    Tensor prevAction = stepInterface.action();
    Tensor nextState = stepInterface.nextState();
    // ---
    Scalar reward = monteCarloInterface.reward(prevState, prevAction, nextState);
    // ---
    Scalar alpha = learningRate.alpha(stepInterface);
    Scalar alpha_gamma_lambda = Times.of(alpha, gamma_lambda);
    Tensor x = featureMapper.getFeature(StateAction.key(prevState, prevAction));
    Scalar prevQ = w.dot(x).Get();
    Scalar nextQ = evaluate(stepInterface);
    Scalar delta = reward.add(gamma.multiply(nextQ)).subtract(prevQ);
    // eq (12.11)
    z = z.multiply(gamma_lambda) //
        .add(x.multiply(RealScalar.ONE.subtract(alpha_gamma_lambda.multiply(z.dot(x).Get()))));
    // ---
    Scalar diffQ = prevQ.subtract(nextQOld);
    Tensor scalez = z.multiply(alpha.multiply(delta.add(diffQ)));
    Tensor scalex = x.multiply(alpha.multiply(diffQ));
    w = w.add(scalez).subtract(scalex);
    nextQOld = nextQ;
    // ---
    learningRate.digest(stepInterface);
    // ---
    if (monteCarloInterface.isTerminal(nextState))
      resetEligibility();
  }

  /** @param stepInterface
   * @return */
  protected abstract Scalar evaluate(StepInterface stepInterface);

  private void resetEligibility() {
    nextQOld = RealScalar.ZERO;
    /** eligibility trace vector is initialized to zero at the beginning of the
     * episode */
    z = Array.zeros(featureSize);
  }
}
