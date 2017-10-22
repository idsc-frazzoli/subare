// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.function.BinaryOperator;
import java.util.function.Function;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.red.Max;

public enum DiscreteUtils {
  ;
  /** @param discreteModel
   * @return index for state-action */
  public static Index build(DiscreteModel discreteModel, Tensor states) {
    Tensor tensor = Tensors.empty();
    for (Tensor state : states)
      for (Tensor action : discreteModel.actions(state))
        tensor.append(Tensors.of(state, action));
    return Index.build(tensor);
  }

  // ---
  /** @param discreteModel
   * @param qsa
   * @param binaryOperator
   * @return */
  public static DiscreteVs reduce( //
      DiscreteModel discreteModel, QsaInterface qsa, BinaryOperator<Scalar> binaryOperator) {
    return DiscreteVs.build(discreteModel, //
        Tensor.of(discreteModel.states().flatten(0) //
            .map(state -> discreteModel.actions(state).flatten(0) //
                .map(action -> qsa.value(state, action)) //
                .reduce(binaryOperator).get()))); // <- assumes greedy policy
  }

  /** compute state value function v(s) based on given action-value function q(s,a)
   * 
   * @param discreteModel
   * @param qsa
   * @return state values */
  public static DiscreteVs createVs(DiscreteModel discreteModel, QsaInterface qsa) {
    return reduce(discreteModel, qsa, Max::of);
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
