// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.util.FairArgMax;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.TensorRuntimeException;
import ch.ethz.idsc.tensor.sca.Log;
import ch.ethz.idsc.tensor.sca.Sign;
import ch.ethz.idsc.tensor.sca.Sqrt;

/** upper confidence bound is greedy except that it encourages
 * exploration if an action has not been encountered often relative to other actions
 * 
 * p.37 equation (2.8) */
public class UcbPolicy implements Policy, StepDigest {
  /** @param qsa
   * @param c factor for scaling relative to values in qsa
   * @return */
  public static UcbPolicy of(DiscreteModel discreteModel, QsaInterface qsa, Scalar c) {
    return new UcbPolicy(discreteModel, qsa, c);
  }

  // ---
  private final DiscreteModel discreteModel;
  private final QsaInterface qsa;
  private final Scalar c; // TODO can merge c and t into "just t"
  // ---
  private final Map<Tensor, Integer> map = new HashMap<>();
  // ---
  private Scalar t = null;

  private UcbPolicy(DiscreteModel discreteModel, QsaInterface qsa, Scalar c) {
    this.discreteModel = discreteModel;
    this.qsa = qsa;
    this.c = c;
  }

  /** @param t positive */
  public void setTime(Scalar t) {
    if (!Sign.isPositive(t))
      throw TensorRuntimeException.of(t);
    this.t = t;
  }

  // TODO very private and not very efficient -> precompute!!!
  private Scalar valueWithBias(Tensor state, Tensor action) {
    Tensor key = StateAction.key(state, action);
    Scalar Nta = RealScalar.of(map.containsKey(key) ? map.get(key) : 0);
    final Scalar bias;
    if (Scalars.isZero(Nta))
      // if an action hasn't been taken yet, bias towards this action is infinite
      bias = DoubleScalar.POSITIVE_INFINITY;
    else
      bias = c.multiply(Sqrt.of(Log.of(t).divide(Nta))); // p.37 equation (2.8)
    return qsa.value(state, action).add(bias);
  }

  @Override // from PolicyInterface
  public Scalar probability(Tensor state, Tensor action) {
    Tensor actions = discreteModel.actions(state);
    Tensor values = Tensor.of(actions.flatten(0).map(a -> valueWithBias(state, a)));
    FairArgMax fairArgMax = FairArgMax.of(values);
    List<Integer> options = fairArgMax.options();
    Index index = Index.build(actions);
    int input = index.of(action);
    if (options.contains(input))
      return RationalScalar.of(1, fairArgMax.optionsCount());
    return RealScalar.ZERO;
  }

  @Override // from StepDigest
  public void digest(StepInterface stepInterface) {
    // TODO code redundant... find a more elegant solution to count pairs
    Tensor key = StateAction.key(stepInterface);
    int index = map.containsKey(key) ? map.get(key) : 0;
    map.put(key, index + 1);
  }
}
