// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.HashMap;
import java.util.Map;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.TensorRuntimeException;
import ch.ethz.idsc.tensor.sca.Log;
import ch.ethz.idsc.tensor.sca.Sqrt;

/** upper confidence bound is greedy except that it encourages
 * exploration if an action has not been encountered often relative to other actions
 * 
 * p.37 equation (2.8) */
public class UcbPolicy implements PolicyInterface, StepDigest {
  /** @param qsa
   * @param c factor for scaling relative to values in qsa
   * @return */
  public static UcbPolicy of(QsaInterface qsa, Scalar c) {
    return new UcbPolicy(qsa, c);
  }

  // ---
  private final QsaInterface qsa;
  private final Scalar c; // TODO can merge c and t into "just t"
  // ---
  private final Map<Tensor, Integer> map = new HashMap<>();
  // ---
  private Scalar t = null;

  private UcbPolicy(QsaInterface qsa, Scalar c) {
    this.qsa = qsa;
    this.c = c;
  }

  /** @param t positive */
  public void setTime(Scalar t) {
    if (Scalars.lessEquals(t, RealScalar.ZERO))
      throw TensorRuntimeException.of(t);
    this.t = t;
  }

  @Override // from PolicyInterface
  public Scalar policy(Tensor state, Tensor action) {
    Tensor key = DiscreteQsa.createKey(state, action);
    Scalar Nta = RealScalar.of(map.containsKey(key) ? map.get(key) : 0);
    final Scalar bias;
    if (Scalars.isZero(Nta))
      // if an action hasn't been taken yet, bias towards this action is infinite
      bias = RealScalar.POSITIVE_INFINITY;
    else
      bias = c.multiply(Sqrt.of(Log.of(t).divide(Nta))); // p.37 equation (2.8)
    return qsa.value(state, action).add(bias);
  }

  @Override // from StepDigest
  public void digest(StepInterface stepInterface) {
    // TODO code redundant... find a more elegant solution to count pairs
    Tensor state0 = stepInterface.prevState();
    Tensor action = stepInterface.action();
    Tensor key = DiscreteQsa.createKey(state0, action);
    int index = map.containsKey(key) ? map.get(key) : 0;
    map.put(key, index + 1);
  }
}
