public virtual ListSpeechSynthesisTasksResponse ListSpeechSynthesisTasks(ListSpeechSynthesisTasksRequest request){var options = new InvokeOptions();options.RequestMarshaller = ListSpeechSynthesisTasksRequestMarshaller.Instance;options.ResponseUnmarshaller = ListSpeechSynthesisTasksResponseUnmarshaller.Instance;return Invoke<ListSpeechSynthesisTasksResponse>(request, options);}
public virtual UpdateJourneyStateResponse UpdateJourneyState(UpdateJourneyStateRequest request){var options = new InvokeOptions();options.RequestMarshaller = UpdateJourneyStateRequestMarshaller.Instance;options.ResponseUnmarshaller = UpdateJourneyStateResponseUnmarshaller.Instance;return Invoke<UpdateJourneyStateResponse>(request, options);}
public void RemovePresentationFormat(){MutableSection s = (MutableSection)FirstSection;s.RemoveProperty(PropertyIDMap.PID_PRESFORMAT);}
public CellRangeAddressList(int firstRow, int lastRow, int firstCol, int lastCol): this(){AddCellRangeAddress(firstRow, firstCol, lastRow, lastCol);}
public virtual void delete(int key){int i = binarySearch(mKeys, 0, mSize, key);if (i >= 0){if (mValues[i] != DELETED){mValues[i] = DELETED;mGarbage = true;}}}
public virtual NGit.Api.CreateBranchCommand SetStartPoint(RevCommit startPoint){CheckCallable();this.startCommit = startPoint;this.startPoint = null;return this;}
public int centerX(){return (left + right) >> 1;}
public virtual ListPresetsResponse ListPresets(){return ListPresets(new ListPresetsRequest());}
public virtual DeleteFolderContentsResponse DeleteFolderContents(DeleteFolderContentsRequest request){var options = new InvokeOptions();options.RequestMarshaller = DeleteFolderContentsRequestMarshaller.Instance;options.ResponseUnmarshaller = DeleteFolderContentsResponseUnmarshaller.Instance;return Invoke<DeleteFolderContentsResponse>(request, options);}
public virtual GetConsoleOutputResponse GetConsoleOutput(GetConsoleOutputRequest request){var options = new InvokeOptions();options.RequestMarshaller = GetConsoleOutputRequestMarshaller.Instance;options.ResponseUnmarshaller = GetConsoleOutputResponseUnmarshaller.Instance;return Invoke<GetConsoleOutputResponse>(request, options);}
public virtual PutMailboxPermissionsResponse PutMailboxPermissions(PutMailboxPermissionsRequest request){var options = new InvokeOptions();options.RequestMarshaller = PutMailboxPermissionsRequestMarshaller.Instance;options.ResponseUnmarshaller = PutMailboxPermissionsResponseUnmarshaller.Instance;return Invoke<PutMailboxPermissionsResponse>(request, options);}
public virtual DisableSnapshotCopyResponse DisableSnapshotCopy(DisableSnapshotCopyRequest request){var options = new InvokeOptions();options.RequestMarshaller = DisableSnapshotCopyRequestMarshaller.Instance;options.ResponseUnmarshaller = DisableSnapshotCopyResponseUnmarshaller.Instance;return Invoke<DisableSnapshotCopyResponse>(request, options);}
public static string StripExtension(string filename){int idx = filename.IndexOf('.');if (idx != -1){filename = filename.Substring(0, idx);}return filename;}
public override java.nio.ByteBuffer putInt(int value){throw new System.NotImplementedException();}
public int LastIndexOf(int o){int rval = _limit - 1;for (; rval >= 0; rval--){if (o == _array[rval]){break;}}return rval;}
public virtual void SetCountsByTime(int[] counts, long msecStep){countsByTime = counts;countsByTimeStepMSec = msecStep;}
public FeatHdrRecord(RecordInputStream in1){futureHeader = new FtrHeader(in1);isf_sharedFeatureType = in1.ReadShort();reserved = (byte)in1.ReadByte();cbHdrData = in1.ReadInt();rgbHdrData = in1.ReadRemainder();}
public CopyOnWriteArrayList(){elements = libcore.util.EmptyArray.OBJECT;}
public WriteRequest(DeleteRequest deleteRequest){_deleteRequest = deleteRequest;}
public void ReadFully(byte[] buf){_in.ReadFully(buf);}
