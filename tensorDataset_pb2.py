# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: TensorDataset.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import bcl_pb2 as bcl__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='TensorDataset.proto',
  package='',
  syntax='proto2',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x13TensorDataset.proto\x1a\tbcl.proto\"V\n\x0fTensorDataFrame\x12\x14\n\tTimestamp\x18\x01 \x01(\x03:\x01\x30\x12\x16\n\x0bTemperature\x18\x02 \x01(\x05:\x01\x30\x12\x15\n\nSensorData\x18\x03 \x01(\x05:\x01\x30\"\xb6\x01\n\rTensorDataset\x12%\n\x0eStartTimestamp\x18\x01 \x01(\x0b\x32\r.bcl.DateTime\x12&\n\x0f\x46inishTimestamp\x18\x02 \x01(\x0b\x32\r.bcl.DateTime\x12\x1b\n\x0cIsConsistent\x18\x03 \x01(\x08:\x05\x66\x61lse\x12\x12\n\x07TrainId\x18\x04 \x01(\x03:\x01\x30\x12%\n\x04\x44\x61ta\x18\x05 \x03(\x0b\x32\x17.TensorSensorTimeseries\"i\n\x16TensorSensorTimeseries\x12\x14\n\tBoxNumber\x18\x01 \x01(\x05:\x01\x30\x12\x17\n\x0cSensorNumber\x18\x02 \x01(\x05:\x01\x30\x12 \n\x06\x46rames\x18\x03 \x03(\x0b\x32\x10.TensorDataFrame'
  ,
  dependencies=[bcl__pb2.DESCRIPTOR,])




_TENSORDATAFRAME = _descriptor.Descriptor(
  name='TensorDataFrame',
  full_name='TensorDataFrame',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='Timestamp', full_name='TensorDataFrame.Timestamp', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='Temperature', full_name='TensorDataFrame.Temperature', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='SensorData', full_name='TensorDataFrame.SensorData', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=34,
  serialized_end=120,
)


_TENSORDATASET = _descriptor.Descriptor(
  name='TensorDataset',
  full_name='TensorDataset',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='StartTimestamp', full_name='TensorDataset.StartTimestamp', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='FinishTimestamp', full_name='TensorDataset.FinishTimestamp', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='IsConsistent', full_name='TensorDataset.IsConsistent', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='TrainId', full_name='TensorDataset.TrainId', index=3,
      number=4, type=3, cpp_type=2, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='Data', full_name='TensorDataset.Data', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=123,
  serialized_end=305,
)


_TENSORSENSORTIMESERIES = _descriptor.Descriptor(
  name='TensorSensorTimeseries',
  full_name='TensorSensorTimeseries',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='BoxNumber', full_name='TensorSensorTimeseries.BoxNumber', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='SensorNumber', full_name='TensorSensorTimeseries.SensorNumber', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='Frames', full_name='TensorSensorTimeseries.Frames', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=307,
  serialized_end=412,
)

_TENSORDATASET.fields_by_name['StartTimestamp'].message_type = bcl__pb2._DATETIME
_TENSORDATASET.fields_by_name['FinishTimestamp'].message_type = bcl__pb2._DATETIME
_TENSORDATASET.fields_by_name['Data'].message_type = _TENSORSENSORTIMESERIES
_TENSORSENSORTIMESERIES.fields_by_name['Frames'].message_type = _TENSORDATAFRAME
DESCRIPTOR.message_types_by_name['TensorDataFrame'] = _TENSORDATAFRAME
DESCRIPTOR.message_types_by_name['TensorDataset'] = _TENSORDATASET
DESCRIPTOR.message_types_by_name['TensorSensorTimeseries'] = _TENSORSENSORTIMESERIES
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TensorDataFrame = _reflection.GeneratedProtocolMessageType('TensorDataFrame', (_message.Message,), {
  'DESCRIPTOR' : _TENSORDATAFRAME,
  '__module__' : 'TensorDataset_pb2'
  # @@protoc_insertion_point(class_scope:TensorDataFrame)
  })
_sym_db.RegisterMessage(TensorDataFrame)

TensorDataset = _reflection.GeneratedProtocolMessageType('TensorDataset', (_message.Message,), {
  'DESCRIPTOR' : _TENSORDATASET,
  '__module__' : 'TensorDataset_pb2'
  # @@protoc_insertion_point(class_scope:TensorDataset)
  })
_sym_db.RegisterMessage(TensorDataset)

TensorSensorTimeseries = _reflection.GeneratedProtocolMessageType('TensorSensorTimeseries', (_message.Message,), {
  'DESCRIPTOR' : _TENSORSENSORTIMESERIES,
  '__module__' : 'TensorDataset_pb2'
  # @@protoc_insertion_point(class_scope:TensorSensorTimeseries)
  })
_sym_db.RegisterMessage(TensorSensorTimeseries)


# @@protoc_insertion_point(module_scope)
