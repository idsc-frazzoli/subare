// code by fluric
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.StateActionCounterSupplier;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.TrueOnlineInterface;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.subare.core.util.FeatureQsaAdapter;
import ch.ethz.idsc.subare.core.util.FeatureWeight;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.subare.core.util.PolicyBase;
import ch.ethz.idsc.subare.core.util.StateAction;
import ch.ethz.idsc.subare.core.util.StateActionCounterUtil;
import ch.ethz.idsc.subare.util.Coinflip;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.red.Mean;
import ch.ethz.idsc.tensor.red.Times;
import ch.ethz.idsc.tensor.sca.Clips;

public class DoubleTrueOnlineSarsa implements TrueOnlineInterface, StateActionCounterSupplier {
  private final Coinflip coinflip = Coinflip.fair();
  // ---
  private final MonteCarloInterface monteCarloInterface;
  private final FeatureMapper featureMapper;
  private final LearningRate learningRate;
  private final StateActionCounter sac1;
  private final StateActionCounter sac2;
  private final PolicyBase policy1;
  private final PolicyBase policy2;
  // ---
  private final SarsaEvaluation evaluationType;
  private final Scalar gamma;
  private final Scalar gamma_lambda;
  // ---
  /** feature weight vectors w1 and w2 are a long-term memory, accumulating over the lifetime of the system */
  private FeatureWeight w1;
  private FeatureWeight w2;
  // ---
  private Scalar nextQOld;
  /** eligibility trace z is a short-term memory, typically lasting less time than the length of an episode */
  private Tensor z;

  public DoubleTrueOnlineSarsa( //
      MonteCarloInterface monteCarloInterface, SarsaEvaluation evaluationType, Scalar lambda, //
      FeatureMapper featureMapper, //
      LearningRate learningRate, //
      FeatureWeight w1, FeatureWeight w2, //
      StateActionCounter sac1, StateActionCounter sac2, //
      PolicyBase policy1, PolicyBase policy2) {
    this.monteCarloInterface = monteCarloInterface;
    this.evaluationType = evaluationType;
    this.learningRate = learningRate;
    this.sac1 = sac1;
    this.sac2 = sac2;
    this.policy1 = policy1;
    this.policy2 = policy2;
    this.gamma = monteCarloInterface.gamma();
    this.featureMapper = featureMapper;
    gamma_lambda = Times.of(gamma, Clips.unit().requireInside(lambda));
    this.w1 = w1;
    this.w2 = w2;
    resetEligibility();
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
  @Override
  public final QsaInterface qsaInterface() {
    return new FeatureQsaAdapter(getW(), featureMapper);
  }

  /** faster when only part of the qsa is required */
  private QsaInterface qsaInterface(Tensor w) {
    return new FeatureQsaAdapter(w, featureMapper);
  }

  /** @return policy with respect to (w1 + w2) / 2 and sac1+sac2 */
  public PolicyBase getPolicy() {
    PolicyBase copy = policy1.copyOf(policy1);
    copy.setQsa(qsaInterface());
    copy.setSac(StateActionCounterUtil.getSummedSac(sac1, sac2, monteCarloInterface));
    return copy;
  }

  /** @return unmodifiable weight vector w */
  public final Tensor getW() {
    return Mean.of(Tensors.of(w1.get(), w2.get())).unmodifiable();
  }

  @Override // from StepDigest
  public final void digest(StepInterface stepInterface) {
    policy1.setQsa(qsaInterface(w1.get()));
    policy2.setQsa(qsaInterface(w2.get()));
    // randomly select which w to read and write
    boolean flip = coinflip.tossHead(); // flip coin, probability 0.5 each
    FeatureWeight W1 = flip ? w2 : w1;
    StateActionCounter Sac1 = flip ? sac2 : sac1; // for updating
    PolicyBase Policy1 = flip ? policy1 : policy2;
    PolicyBase Policy2 = flip ? policy2 : policy1;
    // ---
    Tensor prevState = stepInterface.prevState();
    Tensor prevAction = stepInterface.action();
    Tensor nextState = stepInterface.nextState();
    // ---
    Scalar reward = monteCarloInterface.reward(prevState, prevAction, nextState);
    // ---
    Scalar alpha = learningRate.alpha(stepInterface, Sac1);
    Scalar alpha_gamma_lambda = Times.of(alpha, gamma_lambda);
    Tensor x = featureMapper.getFeature(StateAction.key(prevState, prevAction));
    Scalar prevQ = W1.get().dot(x).Get();
    Scalar nextQ = evaluationType.crossEvaluate(nextState, Policy1, Policy2);
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
    Sac1.digest(stepInterface);
    // ---
    if (monteCarloInterface.isTerminal(nextState))
      resetEligibility();
  }

  private final void resetEligibility() {
    nextQOld = RealScalar.ZERO;
    /** eligibility trace vector is initialized to zero at the beginning of the episode */
    z = Array.zeros(featureMapper.featureSize());
  }

  @Override
  public StateActionCounter sac() {
    return StateActionCounterUtil.getSummedSac(sac1, sac2, monteCarloInterface);
  }
}
