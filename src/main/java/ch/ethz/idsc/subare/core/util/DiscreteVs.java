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
import ch.ethz.idsc.tensor.red.Norm;

public class DiscreteVs implements VsInterface {
  /** initializes all state value to zero
   * 
   * @param discreteModel
   * @return */
  public static DiscreteVs build(DiscreteModel discreteModel) {
    Index index = Index.build(discreteModel.states());
    return new DiscreteVs(index, Array.zeros(index.size()));
  }

  public static DiscreteVs build(DiscreteModel discreteModel, Tensor values) {
    return new DiscreteVs(Index.build(discreteModel.states()), values);
  }

  // TODO this assumes greedy policy
  // TODO function does not belong here
  public static DiscreteVs create(DiscreteModel discreteModel, QsaInterface qsa) {
    DiscreteVs discreteVs = build(discreteModel);
    for (Tensor state : discreteModel.states()) {
      Scalar max = discreteModel.actions(state).flatten(0) //
          .map(action -> qsa.value(state, action)) //
          .reduce(Max::of).get();
      discreteVs.increment(state, max); // assumes that initialized to 0
    }
    return discreteVs;
  }

  private final Index index;
  private Tensor values;

  /** @param index
   * @param values */
  public DiscreteVs(Index index, Tensor values) {
    if (index.size() != values.length())
      throw new RuntimeException();
    this.index = index;
    this.values = values;
  }

  @Override
  public Scalar value(Tensor state) {
    return values.Get(index.of(state));
  }

  @Override
  public synchronized void increment(Tensor state, Scalar delta) {
    values.set(scalar -> scalar.add(delta), index.of(state));
  }

  @Deprecated
  public void setAll(Tensor values) {
    this.values = values.copy();
  }

  public DiscreteVs copy() {
    return new DiscreteVs(index, values.copy());
  }

  public DiscreteVs discounted(Scalar gamma) {
    return new DiscreteVs(index, values.multiply(gamma));
  }

  public Tensor values() {
    return values;
  }

  public static Scalar difference(DiscreteVs d1, DiscreteVs d2) {
    return Norm._1.of(d1.values().subtract(d2.values()));
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
}
