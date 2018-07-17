// code by fluric
package ch.ethz.idsc.subare.core.td;

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
  private final Scalar gamma;
  protected final FeatureMapper featureMapper;
  protected final LearningRate learningRate;
  // ---
  private final Scalar gamma_lambda;
  protected Scalar epsilon;
  private final int featureSize;
  // ---
  /** weight vector w is a long-term memory, accumulating over the lifetime of the system */
  protected Tensor w;
  private Scalar nextQOld;
  /** eligibility trace z is a short-term memory, typically lasting less time than the length of an episode */
  private Tensor z;

  /** @param monteCarloInterface
   * @param lambda in [0, 1] Figure 12.14 in the book suggests that lambda in [0.8, 0.9]
   * tends to be a good choice
   * @param learningRate
   * @param featureMapper */
  protected TrueOnlineSarsa(MonteCarloInterface monteCarloInterface, Scalar lambda, LearningRate learningRate, FeatureMapper featureMapper, Tensor w) {
    this.monteCarloInterface = monteCarloInterface;
    this.learningRate = learningRate;
    Clip.unit().requireInside(lambda);
    this.gamma = monteCarloInterface.gamma();
    this.featureMapper = featureMapper;
    gamma_lambda = Times.of(gamma, lambda);
    featureSize = featureMapper.featureSize();
    this.w = w;
    if (w == null)
      this.w = Array.zeros(featureSize);
    resetEligibility();
  }

  /** @param epsilon in [0,1] */
  public final void setExplore(Scalar epsilon) {
    Clip.unit().requireInside(epsilon);
    this.epsilon = epsilon;
  }

  public void printValues() {
    System.out.println("Values for all state-action pairs:");
    for (Tensor state : monteCarloInterface.states())
      for (Tensor action : monteCarloInterface.actions(state)) {
        Tensor stateActionPair = StateAction.key(state, action);
        System.out.println(state + " -> " + action + " " + featureMapper.getFeature(stateActionPair).dot(w));
      }
  }

  /** Returns the Qsa according to the current feature weights.
   * Only use this function, when the state-action space is small enough. */
  // from DiscreteQsaSupplier
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

  public Tensor getW() {
    return w.unmodifiable();
  }

  @Override
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
    Scalar nextQ = evalute(stepInterface);
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
    if (monteCarloInterface.isTerminal(nextState)) {
      resetEligibility();
    }
  }

  protected abstract Scalar evalute(StepInterface stepInterface);

  private void resetEligibility() {
    nextQOld = RealScalar.ZERO;
    /** eligibility trace vector is initialized to zero at the beginning of the
     * episode */
    z = Array.zeros(featureSize);
  }

  public LearningRate getLearningRate() {
    return learningRate;
  }

  public FeatureMapper getFeatureMapper() {
    return featureMapper;
  }
}
