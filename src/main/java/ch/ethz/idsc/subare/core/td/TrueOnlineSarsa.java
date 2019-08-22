// code by fluric
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.subare.core.util.FeatureQsaAdapter;
import ch.ethz.idsc.subare.core.util.FeatureWeight;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.subare.core.util.PolicyBase;
import ch.ethz.idsc.subare.core.util.PolicyExt;
import ch.ethz.idsc.subare.core.util.StateAction;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.red.Times;

/** implementation of box "True Online Sarsa(lambda) for estimating w'x approx. q_pi or q_*
 * 
 * the evaluation types {@link Expected Reward} and {@link Q-Learning} are adapted from the original Sarsa
 * 
 * https://github.com/idsc-frazzoli/subare/files/2257720/trueOnlineSarsa.pdf
 * 
 * in Section 12.8, p.309 */
public class TrueOnlineSarsa extends AbstractTrueOnlineSarsa {
  private final PolicyExt policy;
  /** feature weight vector w is a long-term memory, accumulating over the lifetime of the system */
  private final FeatureWeight w;
  private final StateActionCounter sac;
  private final int featureSize;
  // ---
  /** eligibility trace z is a short-term memory, typically lasting less time than the length of an episode */
  private Tensor z;
  private Scalar nextQOld;

  public TrueOnlineSarsa( //
      MonteCarloInterface monteCarloInterface, SarsaEvaluation evaluationType, //
      Scalar lambda, FeatureMapper featureMapper, LearningRate learningRate, //
      FeatureWeight w, StateActionCounter sac, PolicyExt policy) {
    super(monteCarloInterface, evaluationType, lambda, learningRate, featureMapper);
    this.sac = sac;
    this.w = w;
    this.policy = policy;
    featureSize = featureMapper.featureSize();
    resetEligibility();
  }

  /** faster when only part of the qsa is required */
  @Override // from QsaInterfaceSupplier
  public final QsaInterface qsaInterface() {
    return new FeatureQsaAdapter(w.get(), featureMapper);
  }

  /** @return unmodifiable weight vector w */
  @Override
  public final Tensor getW() {
    return w.get().unmodifiable();
  }

  @Override // from StepDigest
  public final void digest(StepInterface stepInterface) {
    ((PolicyBase) policy).setQsa(qsaInterface());
    Tensor prevState = stepInterface.prevState();
    Tensor prevAction = stepInterface.action();
    Tensor nextState = stepInterface.nextState();
    Scalar reward = stepInterface.reward();
    // ---
    Scalar alpha = learningRate.alpha(stepInterface, sac);
    Scalar alpha_gamma_lambda = Times.of(alpha, gamma_lambda);
    Tensor x = featureMapper.getFeature(StateAction.key(prevState, prevAction));
    Scalar prevQ = (Scalar) w.get().dot(x);
    Scalar nextQ = evaluationType.evaluate(nextState, policy);
    Scalar delta = reward.add(gamma.multiply(nextQ)).subtract(prevQ);
    // eq (12.11)
    z = z.multiply(gamma_lambda) //
        .add(x.multiply(RealScalar.ONE.subtract(z.dot(x).multiply(alpha_gamma_lambda))));
    // ---
    Scalar diffQ = prevQ.subtract(nextQOld);
    Tensor scalez = z.multiply(alpha.multiply(delta.add(diffQ)));
    Tensor scalex = x.multiply(alpha.multiply(diffQ));
    w.set(w.get().add(scalez).subtract(scalex));
    nextQOld = nextQ;
    // ---
    sac.digest(stepInterface);
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

  @Override
  public StateActionCounter sac() {
    return sac;
  }
}
