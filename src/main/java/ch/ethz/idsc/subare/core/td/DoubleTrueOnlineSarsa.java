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
import ch.ethz.idsc.subare.core.util.FeatureWeight;
import ch.ethz.idsc.subare.core.util.StateAction;
import ch.ethz.idsc.subare.util.Coinflip;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.red.Mean;
import ch.ethz.idsc.tensor.red.Times;
import ch.ethz.idsc.tensor.sca.Clip;

public class DoubleTrueOnlineSarsa implements DiscreteQsaSupplier, StepDigest {
  private final Coinflip COINFLIP = Coinflip.fair();
  // ---
  private final MonteCarloInterface monteCarloInterface;
  private final FeatureMapper featureMapper;
  private final LearningRate learningRate1;
  private final LearningRate learningRate2;
  // ---
  private final SarsaEvaluation evaluationType;
  private final Scalar gamma;
  private final Scalar gamma_lambda;
  private final int featureSize;
  // ---
  private Scalar epsilon;
  /** feature weight vectors w1 and w2 are a long-term memory, accumulating over the lifetime of the system */
  private FeatureWeight w1;
  private FeatureWeight w2;
  // ---
  private Scalar nextQOld;
  /** eligibility trace z is a short-term memory, typically lasting less time than the length of an episode */
  private Tensor z;

  /* package */ DoubleTrueOnlineSarsa(MonteCarloInterface monteCarloInterface, SarsaEvaluation evaluationType, Scalar lambda, FeatureMapper featureMapper,
      LearningRate learningRate1, LearningRate learningRate2, FeatureWeight w1, FeatureWeight w2) {
    this.monteCarloInterface = monteCarloInterface;
    this.evaluationType = evaluationType;
    this.learningRate1 = learningRate1;
    this.learningRate2 = learningRate2;
    Clip.unit().requireInside(lambda);
    this.gamma = monteCarloInterface.gamma();
    this.featureMapper = featureMapper;
    gamma_lambda = Times.of(gamma, lambda);
    featureSize = featureMapper.featureSize();
    this.w1 = w1;
    this.w2 = w2;
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
        qsa.assign(state, action, featureMapper.getFeature(stateActionPair).dot(getW()).Get());
      }
    return qsa;
  }

  /** faster when only part of the qsa is required */
  public final QsaInterface qsaInterface() {
    return new FeatureQsaAdapter(getW(), featureMapper);
  }

  /** faster when only part of the qsa is required */
  public final QsaInterface qsaInterface(Tensor w) {
    return new FeatureQsaAdapter(w, featureMapper);
  }

  /** @return unmodifiable weight vector w */
  public final Tensor getW() {
    return Mean.of(Tensors.of(w1.get(), w2.get())).unmodifiable();
  }

  @Override // from StepDigest
  public final void digest(StepInterface stepInterface) {
    // randomly select which w to read and write
    boolean flip = COINFLIP.tossHead(); // flip coin, probability 0.5 each
    FeatureWeight W1 = flip ? w2 : w1;
    FeatureWeight W2 = flip ? w1 : w2;
    LearningRate learningRate = flip ? learningRate2 : learningRate1; // for updating
    // ---
    Tensor prevState = stepInterface.prevState();
    Tensor prevAction = stepInterface.action();
    Tensor nextState = stepInterface.nextState();
    Tensor nextActions = Tensor.of(monteCarloInterface.actions(nextState).stream() //
        .filter(nextAction -> learningRate.encountered(nextState, nextAction)));
    // ---
    Scalar reward = monteCarloInterface.reward(prevState, prevAction, nextState);
    // ---
    Scalar alpha = learningRate1.alpha(stepInterface);
    Scalar alpha_gamma_lambda = Times.of(alpha, gamma_lambda);
    Tensor x = featureMapper.getFeature(StateAction.key(prevState, prevAction));
    Scalar prevQ = W2.get().dot(x).Get();
    Scalar nextQ = Tensors.isEmpty(nextActions) //
        ? RealScalar.ZERO
        : evaluationType.crossEvaluate(epsilon, nextState, nextActions, qsaInterface(W1.get()), qsaInterface(W2.get()));
    Scalar delta = reward.add(gamma.multiply(nextQ)).subtract(prevQ);
    // eq (12.11)
    z = z.multiply(gamma_lambda) //
        .add(x.multiply(RealScalar.ONE.subtract(alpha_gamma_lambda.multiply(z.dot(x).Get()))));
    // ---
    Scalar diffQ = prevQ.subtract(nextQOld);
    Tensor scalez = z.multiply(alpha.multiply(delta.add(diffQ)));
    Tensor scalex = x.multiply(alpha.multiply(diffQ));
    if (flip)
      w2.set(w2.get().add(scalez).subtract(scalex));
    else
      w1.set(w1.get().add(scalez).subtract(scalex));
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
