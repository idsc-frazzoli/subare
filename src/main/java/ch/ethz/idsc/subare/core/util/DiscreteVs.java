// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.function.Function;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.VsInterface;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.red.Max;

public class DiscreteVs implements VsInterface {
  /** initializes all state value to zero
   * 
   * @param discreteModel
   * @return */
  public static DiscreteVs build(DiscreteModel discreteModel) {
    return new DiscreteVs(Index.build(discreteModel.states()));
  }

  private final Index index;
  private Tensor values;

  private DiscreteVs(Index index) {
    this.index = index;
    values = Array.zeros(index.size());
  }

  @Override
  public Scalar value(Tensor state) {
    return values.Get(index.of(state));
  }

  @Override
  public synchronized void increment(Tensor state, Scalar delta) {
    values.set(scalar -> scalar.add(delta), index.of(state));
  }
  
  public void setAll(Tensor values) {
    this.values = values.copy();
  }
  
  public Tensor values() {
    return values;
  }

  public void print() {
    print(Function.identity());
  }

  public void print(Function<Scalar, Scalar> ROUND) {
    for (Tensor key : index.keys()) {
      Scalar value = values.Get(index.of(key));
      System.out.println(key + " " + value.map(ROUND));
    }
  }

  public static DiscreteVs create(DiscreteModel discreteModel, QsaInterface qsa) {
    DiscreteVs discreteVs = build(discreteModel);
    for (Tensor state : discreteModel.states()) {
      Scalar max = discreteModel.actions(state).flatten(0) //
          .map(action -> qsa.value(state, action)) //
          .reduce(Max::of).get();
      discreteVs.increment(state, max);
    }
    return discreteVs;
  }
}
