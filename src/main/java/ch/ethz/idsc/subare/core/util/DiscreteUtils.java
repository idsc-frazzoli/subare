// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.function.BinaryOperator;
import java.util.function.Function;

import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StateActionModel;
import ch.ethz.idsc.subare.core.VsInterface;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.red.Max;

public enum DiscreteUtils {
  ;
  /** collects all possible state-action pairs that can be built with {@code states} according to the {@code stateActionModel}
   * 
   * @param stateActionModel
   * @param states
   * @return index for state-action */
  public static Index build(StateActionModel stateActionModel, Tensor states) {
    Tensor tensor = Tensors.empty();
    for (Tensor state : states)
      for (Tensor action : stateActionModel.actions(state))
        tensor.append(Tensors.of(state, action));
    return Index.build(tensor);
  }

  // ---
  /** @param stateActionModel
   * @param qsa
   * @param binaryOperator
   * @return */
  public static DiscreteVs reduce( //
      StateActionModel stateActionModel, QsaInterface qsa, BinaryOperator<Scalar> binaryOperator) {
    return DiscreteVs.build(stateActionModel.states(), //
        Tensor.of(stateActionModel.states().stream() //
            .map(state -> stateActionModel.actions(state).stream() //
                .map(action -> qsa.value(state, action)) //
                .reduce(binaryOperator).get()))); // <- assumes greedy policy
  }

  /** compute state value function v(s) based on given action-value function q(s,a)
   * 
   * @param stateActionModel
   * @param qsa
   * @return state values */
  public static DiscreteVs createVs(StateActionModel stateActionModel, QsaInterface qsa) {
    return reduce(stateActionModel, qsa, Max::of);
  }

  /**************************************************/
  public static void print(DiscreteQsa qsa, Function<Scalar, Scalar> round) {
    for (Tensor key : qsa.keys()) {
      Scalar value = qsa.value(key.get(0), key.get(1));
      System.out.println(key + " " + value.map(round));
    }
  }

  public static void print(DiscreteQsa qsa) {
    print(qsa, Function.identity());
  }

  /**************************************************/
  public static String infoString(DiscreteQsa qsa) {
    StringBuilder stringBuilder = new StringBuilder();
    stringBuilder.append("#{q(s,a)}=" + qsa.size() + "\n");
    stringBuilder.append("   min(q)=" + qsa.getMin() + "\n");
    stringBuilder.append("   max(q)=" + qsa.getMax() + "\n");
    return stringBuilder.toString().trim();
  }

  /**************************************************/
  public static void print(StateActionModel stateActionModel, VsInterface vs, Function<Scalar, Scalar> round) {
    for (Tensor key : stateActionModel.states()) {
      Scalar value = vs.value(key);
      System.out.println(key + " " + value.map(round));
    }
  }

  public static void print(DiscreteVs vs, Function<Scalar, Scalar> round) {
    for (Tensor key : vs.keys()) {
      Scalar value = vs.value(key);
      System.out.println(key + " " + value.map(round));
    }
  }

  public static void print(DiscreteVs vs) {
    print(vs, Function.identity());
  }
}
